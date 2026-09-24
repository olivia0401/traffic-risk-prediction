"""
Train the **retrospective** casualty-severity models (collision + vehicle +
casualty join). These use post-crash casualty fields, so the scores describe
recorded accidents and do not transfer to ahead-of-time prediction; see
scripts/train_severity_leakfree.py for the honest pre-incident model.
Needs collision, vehicle and casualty CSVs (any year) in --data-dir.
"""
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_and_prepare_data
from src.trainer import SeverityClassifier, compare_models


def main():
    parser = argparse.ArgumentParser(
        description='Train traffic accident severity classification models'
    )
    parser.add_argument(
        '--model',
        choices=['lr', 'rf', 'mlp', 'xgb', 'all'],
        default='all',
        help='Model type to train (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory containing CSV files (default: data)'
    )
    parser.add_argument(
        '--output',
        default='models/severity_model.pkl',
        help='Output path for trained model (default: models/severity_model.pkl)'
    )

    args = parser.parse_args()

    print("="*70)
    print("UK TRAFFIC ACCIDENT SEVERITY CLASSIFICATION")
    print("="*70)

    # Load data
    X, y, feature_names = load_and_prepare_data(args.data_dir)

    if args.model == 'all':
        # Compare all models
        print("\n" + "="*70)
        print("COMPARING ALL MODELS")
        print("="*70)

        results = compare_models(X, y)

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['f1_cv_mean'])
        print(f"\nBest model: {best_model[0]} (F1={best_model[1]['f1_cv_mean']:.3f})")

    else:
        # Train single model
        print("\n" + "="*70)
        print(f"TRAINING {args.model.upper()} MODEL")
        print("="*70)

        classifier = SeverityClassifier(model_type=args.model)
        classifier.train(X, y)

        # Save model
        classifier.save_model(args.output)

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    # main() returns None on success; any failure raises, so Python exits 1.
    sys.exit(main() or 0)
