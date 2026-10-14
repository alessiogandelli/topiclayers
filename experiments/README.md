# experiments

Paper research on real and synthetic corpora: Swiss politicians, RumourEval
2019, COP22, and the synthetic `synth_pol` / `synth_llm` generators. This is
**not** the pytest fixture set — the tiny fixtures used by the test suite live
in `src/topiclayers/examples/`.

Run everything from this directory (`experiments/`), so the relative `posts`,
`cache_dir` and `output_dir` values in the configs resolve as intended.

## Layout

```
experiments/
├── datasets/        raw + converted corpora          (gitignored)
│   ├── swiss/       swiss.csv, swiss_users.jsonl, SwissPoliticians-tweets.json.gz
│   └── rumoureval/  rumoureval.csv, rumoureval_users.jsonl, rumoureval2019.tar.bz2
├── configs/         run configs: swiss.yml, rumoureval.yml, cop22.yml
├── scripts/         run_netpol.py, converters/*
├── reports/         write-ups (*.md)
├── cache/           reusable pipeline artifacts      (gitignored)
├── outputs/         netpol_*.csv and run logs        (gitignored)
└── synth_llm/       self-contained synthetic-corpus experiment
```

## Reproduce

```bash
cd topiclayers/experiments

# 1. convert the raw corpora into the csv_posts schema
../.venv/bin/python scripts/converters/swiss_to_topiclayers.py
../.venv/bin/python scripts/converters/rumoureval_to_topiclayers.py

# 2. run the pipeline  (writes to ../out/<name>)
QDRANT_URL= LABEL_API_BASE= ../.venv/bin/topiclayers run configs/swiss.yml
QDRANT_URL= LABEL_API_BASE= ../.venv/bin/topiclayers run configs/rumoureval.yml

# 3. polarisation analysis  (writes to outputs/netpol_<name>.csv)
../.venv/bin/python scripts/run_netpol.py swiss
```

`configs/cop22.yml` uses the large local corpus at `../data/cop22.json`
(repo-root `data/`, gitignored).

The `synth_llm/` bundle has its own scripts, configs and inputs; see
`synth_llm/TUNING_NOTES.md`. Its reports are written to `reports/`.
