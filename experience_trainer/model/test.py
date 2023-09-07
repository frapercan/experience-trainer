import torch
import torch.nn as nn

tensor = torch.rand([11, 2705])

seq = nn.Sequential(
    nn.Linear(2705, 415),
    nn.BatchNorm1d(415),
    nn.ReLU(),
    nn.Linear(415, 83),
    nn.BatchNorm1d(83),
    nn.ReLU(),
    nn.Linear(83, 5),
    nn.BatchNorm1d(5),
    nn.ReLU(),

)

tensor = seq(tensor)

print(tensor.shape)

# print(resized_tensor.shape)  # Debería imprimir torch.Size([11, 56,
