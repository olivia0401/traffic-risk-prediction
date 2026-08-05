#!/usr/bin/env python3
"""
Train the **leakage-free** collision-severity model.

The original severity pipeline (scripts/train_severity.py) merges casualty and
vehicle records and feeds the model attributes that only exist *after* a
collision — age/sex/class/type of casualty — so its high F1 does not transfer to
any real "predict severity ahead of time" use. This script trains the honest
counterpart: it uses only pre-incident road/environment/time context
(``data_loader.load_leakage_free_severity``) and predicts collision severity, so
the reported macro-F1 is what an ahead-of-time model can actually achieve.

Usage:
    python scripts/train_severity_leakfree.py --model lr      # or rf / xgb
    python scripts/train_severity_leakfree.py --model lr --save
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import load_leakage_free_severity   # noqa: E402
from trainer import SeverityClassifier                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Leakage-free severity training")
    ap.add_argument('--model', default='lr', choices=['lr', 'rf', 'xgb', 'mlp'],
                    help="classifier (default: lr — cheapest, no memory blow-up)")
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--save', action='store_true',
                    help="persist the fitted model to models/severity_leakfree.pkl")
    args = ap.parse_args()

    X, y, feats = load_leakage_free_severity(args.data_dir)
    print(f"\nPre-incident features only ({len(feats)}): {', '.join(feats)}")
    print("Target: collision_severity (0=Fatal, 1=Serious, 2=Slight)\n")

    clf = SeverityClassifier(model_type=args.model)
    clf.train(X, y)   # prints 5-fold macro-F1 / recall

    if args.save:
        clf.save_model('models/severity_leakfree.pkl')


if __name__ == '__main__':
    main()
