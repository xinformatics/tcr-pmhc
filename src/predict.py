
import pandas as pd
import numpy as np
import ntpath
import glob

# PyTorch libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.optim as optim

from sklearn.metrics import accuracy_score


def normalize(dataset_X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(dataset_X[0][24:])
    
    out = []
    for i in range(len(dataset_X)):
        arr = dataset_X[i]
        arr[24:] = scaler.transform(arr[24:])
        out.append(arr)
    out = np.stack(out)
    return out

# Load files
filenames = []
dfs = []
for fp in glob.glob("data/*.csv.gz"):
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
        X = dataiter.next()
        X = X.float()
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        y_pred.append(predicted.item())

    return(y_pred)


# import trained model
model = torch.load("src/model.pt")
model.eval()

y_pred = predict(model, X_test)

print("y_pred", y_pred, end = "\n")

# Write y_true, y_pred to disk
outname = "output_predictions.csv"
print("\nSaving TEST set y_pred to", outname)
df_performance = pd.DataFrame({"Complex": filenames, "Y_pred": y_pred},)
df_performance.to_csv(outname)