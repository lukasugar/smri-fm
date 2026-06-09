# evaluation

Internal evaluation and fine-tuning suite.

## Run a smoke probe

```bash
uv run python -m evaluation.main --config src/evaluation/config/fake_probe.yaml
```

Outputs are written under `runs/evaluation/<name>/`:

- `metrics.json`
- `predictions.csv`
- `head-best.pt`

## Add a task

Add a task class implementing `EvaluationTask`, register it in
`evaluation.tasks.build_task`, and make its datasets return at least:

- `image`
- `target`

Optional keys like `id`, `meta`, `mask`, or `covariates` are allowed.

## Add a backbone

Add a backbone adapter that returns named token sequences such as `cls`, `reg`,
or `patch`. The adapter should not pool tokens; heads own pooling.
