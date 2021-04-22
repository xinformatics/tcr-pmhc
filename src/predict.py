import argparse
import pandas as pd
import numpy as np
import ntpath
import glob
import os
import tempfile
import zipfile

# PyTorch libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.optim as optim


# this line is needed to load in the model
from model import Model, normalize

parser = argparse.ArgumentParser()
parser.add_argument('--input-zip')
args = parser.parse_args()

# Load files
filenames = []
dfs = []
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(args.input_zip) as zip:
        files = [file for file in zip.infolist() if file.filename.endswith(".csv.gz")]
        for file in files:
            zip.extract(file, tmpdir)
    for fp in glob.glob(os.path.join(tmpdir, "*.csv.gz")):
        filenames.append(ntpath.basename(fp))
        dfs.append(pd.read_csv(fp).T)

# ML dataset
dataset_X = np.stack(dfs)
dataset_X = normalize(dataset_X)

X_test = torch.from_numpy(dataset_X)

def predict(model, X):
    model.eval()
    
    # Test set
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    dataiter = iter(loader)

    y_pred = []
    for i in range(len(dataiter)):
        X = dataiter.next()[0]
        X = X.float()
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        y_pred.append(predicted.item())

    return(y_pred)


# import trained model
model = torch.load("src/model.pt")
model.eval()

y_pred = predict(model, X_test)

# Write y_true, y_pred to disk
outname = "output_predictions.csv"
print("\nSaving TEST set y_pred to", outname)
df_performance = pd.DataFrame({"name": filenames, "prediction": y_pred},)
df_performance.to_csv(outname, index=False)

print(open(outname, 'r').read())