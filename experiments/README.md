# Experiments

Run scripts for the paper's measured claims. One script per experiment;
each writes a **timestamped CSV** to `results/` with the full config in
the header. Results that cannot be traced to a config are worthless, and
results files are never overwritten (append a new timestamped one).

Seed everything random and record the seed in the CSV header.

## Layout

- `results/` — timestamped output CSVs (gitignored except `.gitkeep`)
- `onboarding/` — the Phase-6 capability-descriptor onboarding study

## Planned runs (map to CLAUDE.md lifecycle table)

| Stage | Experiment | Output |
|---|---|---|
| Operate | Benchmark accuracy per stratum + abstention precision/recall | `results/benchmark_*.csv` |
| Operate | Serialisation ablation (structured / relational / natural) | `results/ablation_*.csv` |
| Debug | Diagnostic accuracy vs. true failure cause | `results/diagnostics_*.csv` |
| Deploy | Descriptor correction effort vs. authoring from scratch | `onboarding/` |
| Deploy | Map-authoring usability (time, errors, SUS, NASA-TLX) | `results/authoring_*.csv` |
