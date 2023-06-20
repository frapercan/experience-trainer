import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import torch.nn.functional as F

class ResNetDQN(pl.LightningModule):
    def __init__(self, num_actions, lr=0.0001):
        super(ResNetDQN, self).__init__()
        self.model = resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_actions)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, actions = batch

        q_values = self.model(images)
        actions = actions.float()  # Convert actions tensor to float
        loss = F.mse_loss(q_values, actions)
        self.log('train_loss', loss)
        print(loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
