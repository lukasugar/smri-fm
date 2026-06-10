import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inspect_masked_vit_mask_patches.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("inspect_masked_vit_mask_patches", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mean_mask_patch_counts_match_masked_vit_batch_trimming():
    module = load_script_module()
    images = torch.zeros(2, 1, 8, 8, 8)
    images[0, :, 0:4, 0:4, 0:4] = 10.0
    images[0, :, 4:8, 4:8, 4:8] = 10.0
    images[1, :, 0:4, 0:4, 0:4] = 10.0

    no_mask = module.summarize_patch_flow(images, patch_size=(4, 4, 4), mask=None)
    mean_mask = images > images.mean(dim=(1, 2, 3, 4), keepdim=True)
    masked = module.summarize_patch_flow(images, patch_size=(4, 4, 4), mask=mean_mask)

    assert no_mask.total_patches == 8
    assert no_mask.visible_patches_before_trim == [8, 8]
    assert no_mask.tokens_entering_transformer == 8
    assert masked.visible_patches_before_trim == [2, 1]
    assert masked.tokens_entering_transformer == 1
    assert masked.dropped_visible_patches_by_batch_trim == [1, 0]


def test_load_pt_image_uses_first_element(tmp_path):
    module = load_script_module()
    image = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    path = tmp_path / "sample.pt"
    torch.save((image, torch.tensor([123.0])), path)

    loaded = module.load_pt_image(path)

    assert torch.equal(loaded, image)
