import pandas as pd
import numpy as np
import argparse

def benchmark(predictions_file, actuals_file):
    predictions = pd.read_csv(predictions_file).to_dict(orient="records")
    actuals = pd.read_csv(actuals_file).to_dict(orient="records")

    mcc = matthews_corrcoef(actuals, predictions)
    result = {}
    result["mcc"] = mcc
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions')
    parser.add_argument('--actual')
    args = parser.parse_args()
    print("Benchmarks:", benchmark(args.predictions, args.actual))
