"""Persist per-replicate records alongside every aggregate.

Whatever is aggregated is also written out, one row per replicate, next to the aggregate:
paired questions (e.g. a McNemar test between two methods on the same replicates) then remain
answerable without re-running anything.
"""
from __future__ import annotations

import csv
import gzip
import os


def save_records(out_dir: str, name: str, records, key_fields: dict | None = None) -> str:
    """Write one gzipped CSV row per replicate.

    `records` is a sequence of dicts, or of dicts-of-dicts keyed by statistic name (the shape
    `dots.evaluate.summarize_alt` returns); the latter is flattened to one row per
    (replicate, statistic). `key_fields` are constant columns identifying the configuration.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, rec in enumerate(records):
        base = {**(key_fields or {}), "rep": i}
        if isinstance(rec, dict) and rec and all(isinstance(v, dict) for v in rec.values()):
            for stat, vals in rec.items():                 # {stat: {field: value}}
                rows.append({**base, "stat": stat, **vals})
        elif isinstance(rec, dict):                        # {field: value}
            rows.append({**base, **rec})
        elif isinstance(rec, (tuple, list)):               # positional record
            rows.append({**base, **{f"v{j}": v for j, v in enumerate(rec)}})
        else:                                              # scalar
            rows.append({**base, "value": rec})
    if not rows:
        return ""
    path = os.path.join(out_dir, name if name.endswith(".gz") else name + ".gz")
    # Merge with an existing file: read it back, take the union of the fields, and rewrite.
    old = []
    if os.path.exists(path):
        with gzip.open(path, "rt", newline="") as f:
            old = list(csv.DictReader(f))
    # Column names read back are strings; a float key such as 0.05 would otherwise open a
    # second column named "0.05" beside the first.
    allrows = old + [{str(k): v for k, v in r.items()} for r in rows]
    fields = []
    for r in allrows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with gzip.open(path, "wt", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, restval="")
        wr.writeheader()
        wr.writerows(allrows)
    return path
