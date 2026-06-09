from pathlib import Path

from omegaconf import OmegaConf

from evaluation.main import load_config


def test_load_config_merges_overrides(tmp_path):
    cfg_path = Path(tmp_path) / "config.yaml"
    cfg_path.write_text(
        """
name: base
output_dir: runs/evaluation
optimization:
  epochs: 1
""".strip()
    )

    cfg = load_config(cfg_path, ["name=override", "optimization.epochs=2"])

    assert cfg.name == "override"
    assert cfg.optimization.epochs == 2
