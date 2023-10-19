import torch
from torchvision.models import densenet201
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("torchlogs/")
model = densenet201()
writer.add_graph(model, torch.zeros(1, 3, 224, 224))
writer.close()