# Codex Environment Setup

Use the following setup commands in Codex/local shell:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Notes

- In the current Codex environment, package installation may fail (especially `numpy`) due to network or package-index restrictions.
- In Codex, this repo is mainly checked via `py_compile`, import checks, and static review.
- If there is no Gurobi license, do **not** run OPF-based dataset generation/evaluation in Codex.
- Long training and OPF-heavy runs should be executed in your local environment with full dependencies and a valid Gurobi license.
- v14/v15 default to `DATASET_CACHE_MODE="load_only"`.
