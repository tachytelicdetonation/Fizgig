import pytest
import torch
import torch.nn.functional as F

from fizgig.networks.lora import DeLoRAModule, LoRAModule, create_network, delora_export_state_dict


def peft_delora_delta(x, down, up, lam, r, w_norm):
    """Forward term of PEFT DeloraLinear (peft/src/peft/tuners/delora/layer.py @b8674c8)."""
    h = F.linear(x * w_norm, down)
    an = torch.clamp(down.norm(dim=1), min=1e-4)
    bn = torch.clamp(up.norm(dim=0), min=1e-4)
    return F.linear(h * ((lam / r) / (an * bn)), up)


def attach(module_class=DeLoRAModule, rank=4, alpha=4.0, **kwargs):
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 12)
    module = module_class("lora_unet_test", base, 1.0, rank, alpha, **kwargs)
    module.apply_to()
    return base, module


def trained(module):
    with torch.no_grad():
        module.lora_up.weight.normal_()
        module.lora_down.weight.mul_(torch.linspace(0.05, 5.0, module.lora_down.weight.shape[1]))
        if hasattr(module, "delora_lambda"):
            module.delora_lambda.fill_(7.0)
    return module


@pytest.mark.parametrize("shape", [(3, 16), (2, 5, 16)])
def test_forward_matches_peft(shape):
    base, module = attach()
    trained(module)
    x = torch.randn(*shape)
    want = module.org_forward(x) + peft_delora_delta(
        x, module.lora_down.weight, module.lora_up.weight, module.delora_lambda, 4, module.delora_w_norm)
    torch.testing.assert_close(base(x), want)


def test_weight_norm_is_the_fixed_column_norm_of_the_base_weight_unless_given():
    base, module = attach()
    torch.testing.assert_close(module.delora_w_norm, base.weight.detach().norm(dim=0))
    given = torch.full((16,), 2.0)
    _, module2 = attach(w_norms={"lora_unet_test": given})
    torch.testing.assert_close(module2.delora_w_norm, given)
    assert "delora_w_norm" in module.state_dict() and not module.delora_w_norm.requires_grad


def test_starts_as_the_base_layer_and_lambda_a_b_all_get_gradients():
    base, module = attach()
    x = torch.randn(3, 16)
    torch.testing.assert_close(base(x), module.org_forward(x))
    with torch.no_grad():
        module.lora_up.weight.normal_()
    base(x).pow(2).sum().backward()
    for p in (module.lora_down.weight, module.lora_up.weight, module.delora_lambda):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_output_differs_from_plain_lora_with_the_same_weights():
    base_d, delora = attach()
    base_l, lora = attach(LoRAModule)
    trained(delora)
    lora.load_state_dict({k: v for k, v in delora.state_dict().items() if not k.startswith("delora_")})
    x = torch.randn(3, 16)
    assert not torch.allclose(base_d(x), base_l(x))


def test_export_is_an_ordinary_lora_with_identical_outputs():
    base_d, delora = attach()
    trained(delora)
    exported = delora_export_state_dict(delora.state_dict())
    assert set(exported) == {"lora_down.weight", "lora_up.weight", "alpha"}
    base_l, lora = attach(LoRAModule)
    lora.load_state_dict(exported)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(base_l(x), base_d(x))


def test_unsupported_layers_are_rejected():
    with pytest.raises(ValueError, match="Linear"):
        DeLoRAModule("c", torch.nn.Conv2d(3, 4, 1), 1.0, 4, 4.0)
    with pytest.raises(ValueError, match="split"):
        DeLoRAModule("s", torch.nn.Linear(16, 12), 1.0, 4, 4.0, split_dims=[4, 8])


def test_network_adds_one_lambda_per_module_to_the_lora_parameters():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(3)])

    def count(module_class, kwargs):
        unet = Tiny()
        net = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet, module_class=module_class, module_kwargs=kwargs)
        net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    assert count(LoRAModule, {}) == 3 * (16 * 4 + 4 * 16)
    assert count(DeLoRAModule, {"delora_lambda": 15.0}) == count(LoRAModule, {}) + 3


def test_trainer_checkpoints_are_exports_and_resume_state_is_raw(tmp_path):
    trainer = pytest.importorskip("fizgig.krea2.trainer", reason="needs the trainer's dependencies (cv2, transformers)")
    from safetensors import safe_open
    from safetensors.torch import load_file

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(2)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    torch.manual_seed(0)
    unet = Tiny()
    net = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet, module_class=DeLoRAModule, module_kwargs={"delora_lambda": 15.0})
    net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    for m in net.unet_loras:
        trained(m)
    net._network_type = "delora"
    x = torch.randn(3, 16)
    live = unet(x).detach()
    ckpt, state = tmp_path / "ckpt.safetensors", tmp_path / "state.safetensors"
    trainer._save_lora(net, str(ckpt), 4, 4.0, torch.float32)
    trainer._save_lora(net, str(state), 4, 4.0, torch.float32, raw=True)
    with safe_open(str(ckpt), "pt") as f:
        assert f.metadata()["ss_adapter_weights"].startswith("exported")
        assert not [k for k in f.keys() if "delora_" in k]
    raw = load_file(str(state))
    for k, v in net.state_dict().items():
        torch.testing.assert_close(raw[k], v)
    torch.manual_seed(0)
    unet2 = Tiny()
    plain = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet2)
    plain.apply_to(None, unet2, apply_text_encoder=False, apply_unet=True)
    plain.load_state_dict(load_file(str(ckpt)))
    torch.testing.assert_close(unet2(x), live)


def test_base_column_norms_are_read_from_the_unquantized_checkpoint_under_lora_names(tmp_path):
    trainer = pytest.importorskip("fizgig.krea2.trainer", reason="needs the trainer's dependencies (cv2, transformers)")
    from safetensors.torch import save_file

    w1, w2 = torch.randn(12, 16), torch.randn(8, 12)
    path = tmp_path / "raw.safetensors"
    save_file({"blocks.0.attn.wq.weight": w1, "blocks.0.attn.wq.bias": torch.randn(12),
               "diffusion_model.txtfusion.projector.weight": w2}, str(path))
    norms = trainer._base_column_norms(str(path))
    assert set(norms) == {"lora_unet_blocks_0_attn_wq", "lora_unet_txtfusion_projector"}
    torch.testing.assert_close(norms["lora_unet_blocks_0_attn_wq"], w1.norm(dim=0))
    torch.testing.assert_close(norms["lora_unet_txtfusion_projector"], w2.norm(dim=0))
