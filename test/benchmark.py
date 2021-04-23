import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import matthews_corrcoef

def benchmark(predictions_file, actuals_file):
    predictions = pd.read_csv(predictions_file)
    actuals = pd.read_csv(actuals_file)

    mcc = matthews_corrcoef(actuals["actual"], predictions["prediction"])
    result = {}
    result["mcc"] = mcc
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions')
    parser.add_argument('--actual')
    args = parser.parse_args()
    print("Benchmarks:", benchmark(args.predictions, args.actual))
