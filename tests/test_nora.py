import pytest
import torch
import torch.nn.functional as F

from fizgig.networks.lora import LoRAModule, NoRAModule, create_network, nora_export_state_dict, nora_normalize


def author_nora_delta(x, down, up, scale):
    """Forward term of Joluck/NoRA (peft/src/peft/tuners/lora/layer.py @4669c35, use_nora=True)."""
    down_hat = down / (down.norm(dim=0, keepdim=True) + 1e-6)
    return F.linear(F.linear(x, down_hat), up) * scale


def attach(module_class, rank=4, alpha=4.0, split_dims=None, **kwargs):
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 12)
    module = module_class("lora_unet_test", base, 1.0, rank, alpha, split_dims=split_dims, **kwargs)
    module.apply_to()
    return base, module


def trained(module):
    """Give the adapter non-trivial weights, with input columns of very different norms."""
    with torch.no_grad():
        ups = module.lora_up if module.split_dims is not None else [module.lora_up]
        downs = module.lora_down if module.split_dims is not None else [module.lora_down]
        for up in ups:
            up.weight.normal_()
        for down in downs:
            down.weight.mul_(torch.linspace(0.05, 5.0, down.weight.shape[1]))
    return module


@pytest.mark.parametrize("shape", [(3, 16), (2, 5, 16)])
def test_forward_matches_the_author_formula(shape):
    base, module = attach(NoRAModule)
    trained(module)
    x = torch.randn(*shape)
    want = module.org_forward(x) + author_nora_delta(x, module.lora_down.weight, module.lora_up.weight, module.scale)
    torch.testing.assert_close(base(x), want)


def test_output_differs_from_plain_lora_with_the_same_weights():
    base_n, nora = attach(NoRAModule)
    base_l, lora = attach(LoRAModule)
    trained(nora)
    lora.load_state_dict(nora.state_dict())
    x = torch.randn(3, 16)
    assert not torch.allclose(base_n(x), base_l(x))


def test_starts_as_the_base_layer_and_the_down_matrix_gets_gradients_through_the_normalisation():
    base, module = attach(NoRAModule)
    x = torch.randn(3, 16)
    torch.testing.assert_close(base(x), module.org_forward(x))
    with torch.no_grad():
        module.lora_up.weight.normal_()
    base(x).pow(2).sum().backward()
    got = module.lora_down.weight.grad.clone()

    down = module.lora_down.weight.detach().clone().requires_grad_(True)
    (module.org_forward(x) + author_nora_delta(x, down, module.lora_up.weight.detach(), module.scale)).pow(2).sum().backward()
    torch.testing.assert_close(got, down.grad)
    assert got.abs().sum() > 0


def test_export_is_an_ordinary_lora_with_identical_outputs():
    base_n, nora = attach(NoRAModule)
    trained(nora)
    exported = nora_export_state_dict(nora.state_dict(), mode="forward")
    base_l, lora = attach(LoRAModule)
    lora.load_state_dict(exported)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(base_l(x), base_n(x))
    norms = exported["lora_down.weight"].norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)


def test_init_mode_normalises_once_then_trains_as_plain_lora():
    base_n, nora = attach(NoRAModule, mode="init")
    norms = nora.lora_down.weight.norm(dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)
    trained(nora)
    base_l, lora = attach(LoRAModule)
    lora.load_state_dict(nora.state_dict())
    x = torch.randn(3, 16)
    torch.testing.assert_close(base_n(x), base_l(x))
    assert nora_export_state_dict(nora.state_dict(), mode="init") is not None


def test_split_dims_normalise_each_down_matrix():
    base, module = attach(NoRAModule, split_dims=[4, 8])
    trained(module)
    x = torch.randn(3, 16)
    parts = [author_nora_delta(x, d.weight, u.weight, module.scale) for d, u in zip(module.lora_down, module.lora_up)]
    torch.testing.assert_close(base(x), module.org_forward(x) + torch.cat(parts, dim=-1))


def test_conv_layers_and_unknown_modes_are_rejected():
    with pytest.raises(ValueError, match="Linear"):
        NoRAModule("c", torch.nn.Conv2d(3, 4, 1), 1.0, 4, 4.0)
    with pytest.raises(ValueError, match="mode"):
        NoRAModule("l", torch.nn.Linear(4, 4), 1.0, 4, 4.0, mode="sometimes")


def test_network_has_the_same_trainable_parameters_as_lora():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(3)])

    def count(module_class, kwargs):
        unet = Tiny()
        net = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet, module_class=module_class, module_kwargs=kwargs)
        return sum(p.numel() for p in net.parameters() if p.requires_grad), len(net.unet_loras)

    assert count(NoRAModule, {"mode": "forward"}) == count(LoRAModule, {})
    assert count(NoRAModule, {"mode": "forward"})[1] == 3


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
    net = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet, module_class=NoRAModule, module_kwargs={"mode": "forward"})
    net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    for m in net.unet_loras:
        trained(m)
    net._network_type, net._nora_mode = "nora", "forward"
    x = torch.randn(3, 16)
    live = unet(x).detach()

    ckpt, state = tmp_path / "ckpt.safetensors", tmp_path / "state.safetensors"
    trainer._save_lora(net, str(ckpt), 4, 4.0, torch.float32)
    trainer._save_lora(net, str(state), 4, 4.0, torch.float32, raw=True)

    with safe_open(str(ckpt), "pt") as f:
        assert f.metadata()["ss_nora_weights"].startswith("exported")
    raw = load_file(str(state))
    for k, v in net.state_dict().items():
        torch.testing.assert_close(raw[k], v)

    torch.manual_seed(0)
    unet2 = Tiny()
    plain = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet2)
    plain.apply_to(None, unet2, apply_text_encoder=False, apply_unet=True)
    plain.load_state_dict(load_file(str(ckpt)))
    torch.testing.assert_close(unet2(x), live)
