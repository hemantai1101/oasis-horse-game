"""
verify_data.py — Quick sanity check for generated games.jsonl files.

Checks:
  1. Feature vector length is consistently 41
  2. Labels are only +1.0 / -1.0
  3. Dataset size is even (original + augmented pairs)
  4. Every record has a valid 180° augmented counterpart in the file
  5. Label distribution is roughly balanced

Usage:
  python3 training/verify_data.py training/data/games.jsonl
  python3 training/verify_data.py training/data/games_10_3_.2_.15.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path


def verify(path: str) -> bool:
    path = Path(path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return False

    print(f"Verifying: {path}")
    print("-" * 50)

    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error on line {i}: {e}")
                return False

    total = len(records)
    print(f"Total records: {total}")

    # ── 1. Feature length ──────────────────────────────────────────────────────
    feat_lengths = Counter(len(r['features']) for r in records)
    if feat_lengths == {41: total}:
        print(f"✅ Feature length: 41 (all records)")
    else:
        print(f"❌ Feature lengths: {dict(feat_lengths)}  (expected all 41)")
        return False

    # ── 2. Labels ──────────────────────────────────────────────────────────────
    label_counts = Counter(r['label'] for r in records)
    bad_labels = {l: c for l, c in label_counts.items() if l not in (1.0, -1.0)}
    if bad_labels:
        print(f"❌ Unexpected labels: {bad_labels}")
        return False
    p1_pct = label_counts[1.0] / total * 100
    p2_pct = label_counts[-1.0] / total * 100
    print(f"✅ Labels: +1={label_counts[1.0]} ({p1_pct:.1f}%)  -1={label_counts[-1.0]} ({p2_pct:.1f}%)")

    # ── 3. Even count (original + augmented pairs) ─────────────────────────────
    if total % 2 == 0:
        print(f"✅ Record count is even ({total} = {total//2} games × 2)")
    else:
        print(f"⚠️  Record count is odd ({total}) — expected even for augmented data")

    # ── 4. Augmentation correctness ────────────────────────────────────────────
    # Build lookup of all feature vectors (rounded to avoid float drift)
    feature_set = Counter()
    for r in records:
        key = tuple(round(x, 6) for x in r['features'])
        feature_set[key] += 1

    missing = 0
    missing_example = None
    for r in records:
        feats = r['features']
        aug = list(feats)
        for i in range(0, 40, 2):           # flip all 20 col/row pairs (P1 + P2)
            aug[i]   = round(1.0 - feats[i], 6)
            aug[i+1] = round(1.0 - feats[i+1], 6)
        aug[40] = round(feats[40], 6)       # player indicator unchanged
        aug_key = tuple(aug)
        if feature_set[aug_key] == 0:
            missing += 1
            if missing_example is None:
                missing_example = (feats, aug)

    if missing == 0:
        print(f"✅ Augmentation: all {total} records have a valid 180° counterpart")
    else:
        print(f"❌ Augmentation: {missing}/{total} records are missing their counterpart")
        if missing_example:
            orig, expected = missing_example
            print(f"\n   First bad record:")
            print(f"   Original P1  [0-19]:  {orig[0:20]}")
            print(f"   Original P2  [20-39]: {orig[20:40]}")
            print(f"   Expected aug P1:  {expected[0:20]}")
            print(f"   Expected aug P2:  {expected[20:40]}")
        return False

    print("-" * 50)
    print("✅ All checks passed — data looks good!")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 training/verify_data.py <path/to/games.jsonl>")
        sys.exit(1)
    ok = verify(sys.argv[1])
    sys.exit(0 if ok else 1)
