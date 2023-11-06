import argparse

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-m',
        '--model',
        default='./model/',
        help='The model to evaluate (default ./model/)'
    )

    parser.add_argument(
        '-d',
        'data-folder',
        default='./data/',
        help='folder that contains the data files (default ./data/)'
    )
    
    parser.add_argument(
        '-p'
        '--predictions-file',
        default='./predictions.txt'
        help='file to save predictions to (defualt ./predictions.txt)'
        
    )

    python3 $EXPERIMENT_FOLDER"predict.py" \
            --model $EXPERIMENT_FOLDER"model/" \
            --test-data $DATA_FOLDER"dev.tsv" \
            --predictions-outp $EXPERIMENT_FOLDER"dev_predictions.txt"
def predict(model_path: str) -> list[int]:
    pass

def main():
    pass

if __name__ == '__main__':
    main()