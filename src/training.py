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
for fp in glob.glob("../data/*.csv.gz"):
    filenames.append(ntpath.basename(fp))
    dfs.append(pd.read_csv(fp).T)
    
# ML dataset
dataset_X = np.stack(dfs)
dataset_X = normalize(dataset_X)
dataset_y = np.array(["P1" in file for file in filenames], dtype = int)

X_train, y_train = torch.from_numpy(dataset_X[0:3]), torch.from_numpy(dataset_y[0:3])
X_test, y_test = torch.from_numpy(dataset_X[3:]), torch.from_numpy(dataset_y[3:])

assert len(dataset_X) == len(dataset_y)
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)


# ML architecture


class Model(nn.Module):
    def __init__(self, inp1=407, out=2):
        super().__init__()
        self.Dense1 = nn.Linear(inp1, 32)
        self.relu1 = nn.ReLU()
        self.Dense2 = nn.Linear(32, 1)
        self.relu2 = nn.ReLU()
        self.Dense3 = nn.Linear(142, 2)
        self.relu3 = nn.ReLU()
        self.out = nn.Softmax(1)

    def forward(self, x):
        x = self.relu1(self.Dense1(x))
        x = self.relu2(self.Dense2(x))
        x = x.view(-1, 142)
        x = self.relu3(self.Dense3(x))
        x = self.out(x)
        return x

def predict(model, X,  y):
    model.eval()
    
    # Test set
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    dataiter = iter(testloader)

    y_pred = []
    for i in range(len(dataiter)):
        X, y = dataiter.next()
        X = X.float()
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        y_pred.append(predicted.item())

    return(y_pred)

def performance(y_pred, y_true):
    from sklearn.metrics import accuracy_score
    return(accuracy_score(y_test, y_pred))
    

# Load
dataset = torch.utils.data.TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1)
testloader = torch.utils.data.DataLoader(dataset, batch_size=1)

# Setup
model = Model()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
device = torch.device("cpu")

# Train
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()
        
        X, y = data
        X = X.float().requires_grad_(True)

        # forward + backward + optimize
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # print statistics
    running_loss += loss.item()
    if epoch % 5 == 0:    # print every 2000 mini-batches
        print('Epoch %d, loss: %.3f' %
              (epoch + 1, running_loss / 5))
        running_loss = 0.0

print('Finished Training\n')

# Evaluate
y_true = y_train
y_pred = predict(model, X_train, y_train)
acc = performance(y_pred, y_true)
print("Training set accuracy:", acc)
print("y_pred", y_pred, "y_true", y_true, end = "\n")

y_true = y_test
y_pred = predict(model, X_test, y_test)
acc = performance(y_pred, y_true)
print("\nTest set accuracy:", acc)
print("y_pred", y_pred, "y_true", y_true, end = "\n")

# Write y_true, y_pred to disk
outname = "output_predictions.csv"
print("\nSaving TEST set y_true, y_pred to", outname)
df_performance = pd.DataFrame({"Complex":filenames[3:], "Test_y_true":y_true, "Test_y_pred":y_pred},)
df_performance.to_csv(outname)

# Write model to disk for use in predict.py
torch.save(model, "model.pt")
