import pytest
import torch
import torch.nn.functional as F

from fizgig.networks.lora import LoRAModule, TimestepHolder, TLoRAModule, create_network, tlora_rank_mask


def author_mask(timestep, max_timestep, rank, min_rank):
    """_compute_timestep_mask from ControlGenAI/T-LoRA tlora_peft/layer.py @9c95c27."""
    r_eff = int(((max_timestep - timestep) / max_timestep) * (rank - min_rank)) + min_rank
    r_eff = max(min_rank, min(rank, r_eff))
    mask = torch.zeros((1, rank))
    mask[:, :r_eff] = 1.0
    return mask


def attach(rank=8, alpha=8.0, min_rank=1):
    torch.manual_seed(0)
    holder = TimestepHolder()
    base = torch.nn.Linear(16, 12)
    module = TLoRAModule("lora_unet_test", base, 1.0, rank, alpha, min_rank=min_rank, timestep=holder)
    module.apply_to()
    with torch.no_grad():
        module.lora_up.weight.normal_()
    return base, module, holder


@pytest.mark.parametrize("t", [0.0, 0.1234, 0.5, 0.87, 0.999, 1.0])
@pytest.mark.parametrize("rank,min_rank", [(8, 1), (32, 4), (4, 4)])
def test_mask_matches_the_author_on_their_integer_timestep_scale(t, rank, min_rank):
    torch.testing.assert_close(tlora_rank_mask(t, rank, min_rank), author_mask(int(t * 1000), 1000, rank, min_rank))


@pytest.mark.parametrize("shape", [(3, 16), (2, 5, 16), (2, 3, 5, 16)])
def test_forward_matches_the_author_at_every_input_rank(shape):
    base, module, holder = attach()
    holder.set(torch.tensor([0.6, 0.1]))
    x = torch.randn(*shape)
    mask = author_mask(int(0.6 * 1000), 1000, 8, 1)
    want = module.org_forward(x) + F.linear(F.linear(x, module.lora_down.weight) * mask, module.lora_up.weight) * module.scale
    torch.testing.assert_close(base(x), want)


def test_full_rank_at_t_zero_equals_plain_lora_and_min_rank_at_t_one_does_not():
    base_t, tl, holder = attach()
    torch.manual_seed(0)
    base_l = torch.nn.Linear(16, 12)
    lora = LoRAModule("lora_unet_test", base_l, 1.0, 8, 8.0)
    lora.apply_to()
    lora.load_state_dict(tl.state_dict())
    x = torch.randn(3, 16)
    holder.set(torch.tensor([0.0]))
    torch.testing.assert_close(base_t(x), base_l(x))
    holder.set(torch.tensor([1.0]))
    assert not torch.allclose(base_t(x), base_l(x))


def test_forward_without_a_timestep_raises():
    base, module, holder = attach()
    with pytest.raises(RuntimeError, match="timestep"):
        base(torch.randn(3, 16))


def test_the_dit_hook_feeds_the_timestep_from_the_t_keyword():
    class Dit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16)])

        def forward(self, img, context=None, t=None):
            return self.blocks[0](img)

    dit = Dit()
    holder = TimestepHolder()
    net = create_network(None, "lora_unet", 1.0, 8, 8.0, None, [], dit, module_class=TLoRAModule,
                         module_kwargs={"min_rank": 2, "timestep": holder})
    net.apply_to(None, dit, apply_text_encoder=False, apply_unet=True)
    handle = holder.attach(dit)
    dit(img=torch.randn(1, 4, 16), t=torch.tensor([0.25]))
    assert holder.t == pytest.approx(0.25)
    handle.remove()


def test_trainer_saves_raw_weights_marked_as_needing_timestep_masking(tmp_path):
    trainer = pytest.importorskip("fizgig.krea2.trainer", reason="needs the trainer's dependencies (cv2, transformers)")
    from safetensors import safe_open

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16)])

    unet = Tiny()
    net = create_network(None, "lora_unet", 1.0, 8, 8.0, None, [], unet, module_class=TLoRAModule,
                         module_kwargs={"min_rank": 2, "timestep": TimestepHolder()})
    net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    net._network_type, net._tlora_min_rank = "tlora", 2
    path = tmp_path / "ckpt.safetensors"
    trainer._save_lora(net, str(path), 8, 8.0, torch.float32)
    with safe_open(str(path), "pt") as f:
        meta = f.metadata()
        assert meta["ss_adapter"] == "tlora" and meta["ss_tlora_min_rank"] == "2"
        assert "timestep" in meta["ss_adapter_weights"]
