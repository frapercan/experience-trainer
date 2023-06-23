import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import torch.nn.functional as F


class ResNet(pl.LightningModule):
    def __init__(self, num_actions, learning_rate=0.001):
        super(ResNet, self).__init__()


        self.learning_rate = learning_rate
        # ResNet backbone
        self.model = resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        # Actor head
        self.actor_fc = nn.Linear(num_features, num_actions)
        self.actor_softmax = nn.Softmax(dim=1)

        # Critic head
        self.critic_fc = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.model(x)
        features = features.view(features.size(0), -1)

        # Actor forward pass
        actor_output = self.actor_fc(features)
        actor_output = self.actor_softmax(actor_output)

        # Critic forward pass
        critic_output = self.critic_fc(features)
        return actor_output, critic_output

    def training_step(self, batch, batch_idx):
        states, actions, returns = batch
        actions = actions.float()  # Convert actions tensor to Float type
        returns = returns.float()  # Convert returns tensor to Float type
        returns = torch.unsqueeze(returns, dim=-1)
        # Forward pass
        actor_output, critic_output = self.forward(states)

        # Compute actor and critic losses
        actor_loss = torch.nn.CrossEntropyLoss()(actor_output, actions)
        critic_loss = torch.nn.MSELoss()(critic_output, returns)

        # Total loss
        total_loss = torch.abs(actor_loss) + torch.abs(critic_loss)

        # Log the losses
        self.log('actor_loss', actor_loss)
        self.log('critic_loss', critic_loss)
        self.log('total_loss', total_loss)
        self.log('lr',self.learning_rate)
        print(self.learning_rate)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, r = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,10)

        return [optimizer], [scheduler]


import torch
import torch.nn as nn
from torchvision.models import resnet18

class ActorCriticModel(nn.Module):
    def __init__(self, num_actions):
        super(ActorCriticModel, self).__init__()



    def forward(self, x):
        features = self.model(x)
        features = features.view(features.size(0), -1)

        # Actor forward pass
        actor_output = self.actor_fc(features)
        actor_output = self.actor_softmax(actor_output)

        # Critic forward pass
        critic_output = self.critic_fc(features)

        return actor_output, critic_output

# class ResNet(pl.LightningModule):
#     def __init__(self, num_actions, lr=0.001):
#         super(ResNet, self).__init__()
#         self.model = resnet18(pretrained=True)
#         num_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_features, num_actions)
#         self.lr = lr
#         self.loss = torch.nn.MSELoss()
#
#     def forward(self, x):
#         return self.model(x)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         y = y.float()
#         loss = self.loss(logits, y)
#         self.log('train_loss', loss, prog_bar=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.loss(logits, y)
#         self.log('val_loss', loss, prog_bar=True)
#
#     def configure_optimizers(self):
#         return optim.Adam(self.model.parameters(), lr=self.lr)
