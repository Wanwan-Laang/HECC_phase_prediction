#!/usr/bin/env python
"""
Batch HECC phase prediction from listing_comp.csv.
Calls predict_from_formula.py in chunks, parses stdout, writes CSV.
Does not modify predict_from_formula.py.
"""
import csv
import os
import re
import subprocess
import sys

CHUNK = 80  # formulas per subprocess call (avoids long command lines)


def row_to_formula(row):
    elem_frac = [
        ('Nb', float(row['Nb_frac'])),
        ('Ta', float(row['Ta_frac'])),
        ('Ti', float(row['Ti_frac'])),
        ('Zr', float(row['Zr_frac'])),
    ]
    parts = []
    for elem, frac in elem_frac:
        if frac > 1e-9:
            parts.append(f"{elem}{frac:.4f}".rstrip('0').rstrip('.'))
    return ''.join(parts)


def run_chunk(formulas):
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'predict_from_formula.py'),
           '--formula'] + formulas
    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"predict_from_formula.py failed: {result.stderr or result.stdout}")
    ann_line = re.search(r'Prediction\(s\) from ANN:\s*(.+)', result.stdout)
    svm_line = re.search(r'Prediction\(s\) from SVM:\s*(.+)', result.stdout)
    if not ann_line or not svm_line:
        raise RuntimeError("Could not parse predictor output.")
    ann_vals = [float(x) for x in ann_line.group(1).split()]
    svm_vals = [float(x) for x in svm_line.group(1).split()]
    return ann_vals, svm_vals


def main():
    csv_path = 'listing_comp.csv'
    out_path = 'listing_comp_predictions.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    if not os.path.isfile(csv_path):
        print(f"Error: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    if not rows:
        print("No rows in CSV.", file=sys.stderr)
        sys.exit(1)

    ids = [r['composition_id'] for r in rows]
    formulas = [row_to_formula(r) for r in rows]

    all_ann, all_svm = [], []
    for start in range(0, len(formulas), CHUNK):
        chunk_f = formulas[start:start + CHUNK]
        n = len(chunk_f)
        print(f"Running chunk {start // CHUNK + 1} ({n} compositions)...", flush=True)
        ann_vals, svm_vals = run_chunk(chunk_f)
        assert len(ann_vals) == n and len(svm_vals) == n
        all_ann.extend(ann_vals)
        all_svm.extend(svm_vals)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['composition_id', 'formula', 'ANN_multi_phase', 'SVM_multi_phase'])
        for i, cid in enumerate(ids):
            w.writerow([cid, formulas[i], f"{all_ann[i]:.4f}", f"{all_svm[i]:.4f}"])

    print(f"Done. Wrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
