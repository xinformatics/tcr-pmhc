import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--predictions')
parser.add_argument('--actual')
args = parser.parse_args()

predictions = pd.read_csv(args.predictions)
actual = pd.read_csv(args.actual)

