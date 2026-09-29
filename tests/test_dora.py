import pytest
import torch
import torch.nn.functional as F

from fizgig.networks.lora import DoRAModule, LoRAModule, create_network, dora_export_state_dict


def peft_dora_output(x, weight, bias, down, up, magnitude, scale):
    """Base output plus PEFT's DoraLinearLayer term (peft/src/peft/tuners/lora/dora.py @b8674c8), dense weights."""
    weight_norm = (weight + scale * (up @ down)).norm(dim=1).detach()
    mag_norm_scale = (magnitude / weight_norm).view(1, -1)
    base_result = F.linear(x, weight)
    lora_result = F.linear(F.linear(x, down), up)
    return F.linear(x, weight, bias) + (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scale


def comfy_weight_decompose(dora_scale, weight, lora_diff):
    """ComfyUI comfy/weight_adapter/base.py weight_decompose, output-axis branch, strength 1."""
    weight_norm = weight.norm(dim=1, keepdim=True) + torch.finfo(weight.dtype).eps
    return (weight + lora_diff) * (dora_scale / weight_norm)


def attach(module_class=DoRAModule, rank=4, alpha=2.0, **kwargs):
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 12)
    module = module_class("lora_unet_test", base, 1.0, rank, alpha, **kwargs)
    module.apply_to()
    return base, module


def trained(module):
    with torch.no_grad():
        module.lora_up.weight.normal_()
        module.lora_down.weight.mul_(3.0)
        if hasattr(module, "dora_magnitude"):
            module.dora_magnitude.mul_(torch.linspace(0.5, 1.5, module.dora_magnitude.numel()))
    return module


@pytest.mark.parametrize("shape", [(3, 16), (2, 5, 16)])
def test_forward_matches_peft_with_dense_weights(shape):
    base, module = attach()
    trained(module)
    x = torch.randn(*shape)
    want = peft_dora_output(x, base.weight, base.bias, module.lora_down.weight, module.lora_up.weight, module.dora_magnitude, module.scale)
    torch.testing.assert_close(base(x), want)


def test_detached_norm_without_the_dense_weight_equals_the_dense_norm():
    base, module = attach()
    trained(module)
    dense = (base.weight + module.scale * (module.lora_up.weight @ module.lora_down.weight)).norm(dim=1)
    torch.testing.assert_close(module.weight_norm(), dense)


def test_starts_as_the_base_layer_and_magnitude_a_b_get_gradients():
    base, module = attach()
    x = torch.randn(3, 16)
    torch.testing.assert_close(base(x), module.org_forward(x))
    torch.testing.assert_close(module.dora_magnitude.detach(), base.weight.detach().norm(dim=1))
    with torch.no_grad():
        module.lora_up.weight.normal_()
        module.dora_magnitude.mul_(1.2)
    base(x).pow(2).sum().backward()
    for p in (module.lora_down.weight, module.lora_up.weight, module.dora_magnitude):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_output_differs_from_plain_lora_with_the_same_weights():
    base_d, dora = attach()
    base_l, lora = attach(LoRAModule)
    trained(dora)
    lora.load_state_dict({k: v for k, v in dora.state_dict().items() if not k.startswith("dora_")})
    x = torch.randn(3, 16)
    assert not torch.allclose(base_d(x), base_l(x))


def test_comfyui_export_reproduces_the_module_on_its_training_base():
    base, module = attach()
    trained(module)
    exported = dora_export_state_dict(module.state_dict(), {"": module.weight_norm()})
    assert set(exported) == {"lora_down.weight", "lora_up.weight", "alpha", "dora_scale"}
    diff = module.scale * (exported["lora_up.weight"] @ exported["lora_down.weight"])
    merged = comfy_weight_decompose(exported["dora_scale"], base.weight.detach(), diff)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(F.linear(x, merged, base.bias), base(x), atol=1e-5, rtol=1e-4)


def test_unsupported_layers_are_rejected():
    with pytest.raises(ValueError, match="Linear"):
        DoRAModule("c", torch.nn.Conv2d(3, 4, 1), 1.0, 4, 4.0)
    with pytest.raises(ValueError, match="split"):
        DoRAModule("s", torch.nn.Linear(16, 12), 1.0, 4, 4.0, split_dims=[4, 8])
    fp8 = torch.nn.Linear(16, 12)
    fp8.weight.data = fp8.weight.data.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="row_norms"):
        DoRAModule("q", fp8, 1.0, 4, 4.0)


def test_network_adds_one_magnitude_per_output_to_the_lora_parameters():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(3)])

    def count(module_class):
        unet = Tiny()
        net = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet, module_class=module_class)
        net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    assert count(LoRAModule) == 3 * (16 * 4 + 4 * 16)
    assert count(DoRAModule) == count(LoRAModule) + 3 * 16


def test_trainer_saves_comfyui_export_and_raw_resume_state(tmp_path):
    trainer = pytest.importorskip("fizgig.krea2.trainer", reason="needs the trainer's dependencies (cv2, transformers)")
    from safetensors import safe_open
    from safetensors.torch import load_file

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16)])

    torch.manual_seed(0)
    unet = Tiny()
    net = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet, module_class=DoRAModule)
    net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    for m in net.unet_loras:
        trained(m)
    net._network_type = "dora"
    ckpt, state = tmp_path / "ckpt.safetensors", tmp_path / "state.safetensors"
    trainer._save_lora(net, str(ckpt), 4, 4.0, torch.float32)
    trainer._save_lora(net, str(state), 4, 4.0, torch.float32, raw=True)
    with safe_open(str(ckpt), "pt") as f:
        assert f.metadata()["ss_adapter"] == "dora"
        assert any(k.endswith("dora_scale") for k in f.keys())
    raw = load_file(str(state))
    for k, v in net.state_dict().items():
        torch.testing.assert_close(raw[k], v)
