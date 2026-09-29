import pytest
import torch

from fizgig.networks.lora import DoRAModule, LoRAModule, create_network
from fizgig.scripts.krea2_eval import check_weights, load_prompts


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16)])


def network(module_class=LoRAModule):
    unet = Tiny()
    net = create_network(None, "lora_unet", 1.0, 4, 4.0, None, [], unet, module_class=module_class)
    net.apply_to(None, unet, apply_text_encoder=False, apply_unet=True)
    return net


def test_matching_raw_weights_pass_and_mismatches_stop_the_run():
    net = network(DoRAModule)
    raw = net.state_dict()
    check_weights(net, raw, {"ss_adapter_weights": "raw (resume state)"}, "dora")
    with pytest.raises(SystemExit, match="export"):
        check_weights(net, raw, {"ss_adapter_weights": "exported: ComfyUI DoRA"}, "dora")
    missing = {k: v for k, v in raw.items() if not k.endswith("dora_magnitude")}
    with pytest.raises(SystemExit, match="missing"):
        check_weights(net, missing, {}, "dora")
    with pytest.raises(SystemExit, match="unexpected"):
        check_weights(network(), {**network().state_dict(), "stray.weight": torch.zeros(1)}, {}, "lora")


def test_prompts_file_in_the_weeks_format(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"seeds": [1729, 2718], "prompts": [{"id": "p01", "text": "a"}, {"id": "p02", "text": "b"}]}')
    assert load_prompts(str(path)) == ([("p01", "a"), ("p02", "b")], [1729, 2718])
