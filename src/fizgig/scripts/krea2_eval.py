"""Render a fixed prompt x seed grid on the Krea 2 RAW model with one adapter, for adapter comparisons.

The DiT and adapter are built exactly as training builds them (same quantization, same module class), and the
adapter's raw training weights are loaded: the state dir's ``lora.safetensors`` written by
``--save_state_on_train_end``. So DoRA and T-LoRA render through their real modules, which no static LoRA file can
express. ``--network_type none`` renders the unadapted base. The weights must match the network exactly; a
missing or unexpected key stops the run instead of rendering a silent zero adapter.

    python src/fizgig/scripts/krea2_eval.py --dit RAW --vae VAE --text_encoder TE --prompts_json prompts.json \\
        --network_type dora --weights out/run-000010-state/lora.safetensors --out_dir eval/dora
"""

import argparse
import hashlib
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

log = logging.getLogger("krea2_eval")
EXPORTED = "exported"


def load_prompts(path: str) -> tuple[list[tuple[str, str]], list[int]]:
    """Prompts file in the Weeks screening format: {"seeds": [...], "prompts": [{"id", "text"}, ...]}."""
    data = json.load(open(path))
    return [(p["id"], p["text"]) for p in data["prompts"]], [int(s) for s in data["seeds"]]


def check_weights(network, state_dict: dict, metadata: dict, network_type: str) -> None:
    kind = metadata.get("ss_adapter_weights", "")
    if network_type in ("delora", "dora") and kind.startswith(EXPORTED):
        raise SystemExit(f"{network_type}: this file is an export ({kind}); pass the raw state dir's lora.safetensors")
    expected = set(network.state_dict())
    missing, unexpected = sorted(expected - set(state_dict)), sorted(set(state_dict) - expected)
    if missing or unexpected:
        raise SystemExit(f"weights do not match the {network_type} network: missing {missing[:3]} "
                         f"({len(missing)}), unexpected {unexpected[:3]} ({len(unexpected)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit", required=True, help="Krea 2 RAW DiT")
    p.add_argument("--vae", required=True)
    p.add_argument("--text_encoder", required=True)
    p.add_argument("--prompts_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--network_type", required=True, choices=["none", "lora", "lokr", "nora", "delora", "dora", "tlora"])
    p.add_argument("--weights", help="raw adapter weights (not needed for --network_type none)")
    p.add_argument("--network_dim", type=int, default=32)
    p.add_argument("--network_alpha", type=float, default=32)
    p.add_argument("--lokr_factor", type=int, default=8)
    p.add_argument("--nora_mode", default="forward", choices=["forward", "init"])
    p.add_argument("--delora_lambda", type=float, default=15.0)
    p.add_argument("--tlora_min_rank", type=int, default=1)
    p.add_argument("--quant_int8", default="bf16", choices=["", "bf16", "int8"], help="same base profile as training")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--steps", type=int, default=52)
    p.add_argument("--cfg_scale", type=float, default=3.5)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")
    if args.network_type != "none" and not args.weights:
        p.error("--weights is required unless --network_type none")

    from safetensors import safe_open
    from safetensors.torch import load_file
    from fizgig.krea2 import sampling
    from fizgig.krea2.trainer import load_dit_for_training
    from fizgig.krea2.utils import load_krea2_text_encoder
    from fizgig.krea2.vae_loader import load_vae

    prompts, seeds = load_prompts(args.prompts_json)
    os.makedirs(args.out_dir, exist_ok=True)
    enc = load_krea2_text_encoder(args.text_encoder, dtype=torch.bfloat16, device="cuda")
    encoded = {}
    for pid, text in prompts:
        txt, txtmask, untxt, untxtmask = sampling.encode_prompts(enc, [text], cfg=True)
        encoded[pid] = [t.cpu() if t is not None else None for t in (txt, txtmask, untxt, untxtmask)]
    del enc
    torch.cuda.empty_cache()
    ae = load_vae(args.vae, input_channels=3, device="cpu", disable_mmap=True)

    net_type = "lora" if args.network_type == "none" else args.network_type
    dit, network, _, _ = load_dit_for_training(
        args.dit, network_dim=args.network_dim, network_alpha=args.network_alpha, network_type=net_type,
        lokr_factor=args.lokr_factor, nora_mode=args.nora_mode, delora_lambda=args.delora_lambda,
        tlora_min_rank=args.tlora_min_rank, quant_int8=args.quant_int8, gradient_checkpointing=False,
        compile_blocks=False)
    weights_sha = None
    if args.network_type != "none":
        sd = load_file(args.weights)
        with safe_open(args.weights, "pt") as f:
            meta = f.metadata() or {}
        check_weights(network, sd, meta, args.network_type)
        network.load_state_dict(sd, strict=True)
        weights_sha = hashlib.sha256(open(args.weights, "rb").read()).hexdigest()
    network.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    dit.eval()

    records = []
    with torch.no_grad():
        for pid, _text in prompts:
            txt, txtmask, untxt, untxtmask = (t.to("cuda") if t is not None else None for t in encoded[pid])
            for seed in seeds:
                img = sampling.sample(dit, ae, txt, txtmask, untxt=untxt, untxtmask=untxtmask, device="cuda",
                                      dtype=torch.bfloat16, width=args.width, height=args.height, steps=args.steps,
                                      cfg_scale=args.cfg_scale, seed=seed)[0]
                path = os.path.join(args.out_dir, f"{pid}_s{seed}.png")
                img.save(path)
                records.append({"prompt_id": pid, "seed": seed, "file": os.path.basename(path)})
                log.info("%s seed %d -> %s", pid, seed, path)
    manifest = {"network_type": args.network_type, "weights": args.weights, "weights_sha256": weights_sha,
                "settings": {k: v for k, v in vars(args).items() if k not in ("verbose",)}, "images": records}
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2)
    log.info("wrote %d images and manifest.json to %s", len(records), args.out_dir)


if __name__ == "__main__":
    main()
