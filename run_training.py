# run_training.py
import sys
from pathlib import Path

# Add src directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from data_loader import load_and_prepare_data
from trainer import compare_models

def main():
    """
    Main function to run the data loading and model comparison.
    """
    # Load and prepare data from the 'data/' directory
    X, y, feature_names = load_and_prepare_data(data_dir='data')

    # Compare all defined models
    compare_models(X, y)

if __name__ == '__main__':
    main()
