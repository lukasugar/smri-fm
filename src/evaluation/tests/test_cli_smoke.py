from pathlib import Path

from evaluation.main import main


def test_main_runs_fake_probe(tmp_path):
    cfg_path = Path(tmp_path) / "config.yaml"
    cfg_path.write_text(
        f"""
name: fake_probe
output_dir: {tmp_path}
wandb_logging: false
task:
  name: fake_regression
  overwrite_data: false
model:
  name: fake
  embed_dim: 4
mode:
  name: probe
representation: cls
head:
  name: linear
  pooling: first
optimization:
  epochs: 1
  batch_size: 2
  lr: 1e-2
  weight_decay: 0.0
  num_workers: 0
evaluation:
  selection_metric: mae
  selection_mode: min
device: cpu
seed: 7338
""".strip()
    )

    main(cfg_path, [])

    assert (tmp_path / "fake_probe" / "metrics.json").exists()
    assert (tmp_path / "fake_probe" / "predictions.csv").exists()
