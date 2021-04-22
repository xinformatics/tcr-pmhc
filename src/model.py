import torch.nn as nn
import numpy as np

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
