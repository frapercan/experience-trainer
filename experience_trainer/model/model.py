import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import torch.nn.functional as F


class ActorCritic(pl.LightningModule):
    def __init__(self, num_actions, learning_rate=0.001):
        super(ActorCritic, self).__init__()
        self.learning_rate = learning_rate
        resnet = resnet18(pretrained=True)
        self.resnet = resnet
        self.actor = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_actions),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Linear(1005, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.ReLU(),
        )


    def forward(self, x):
        x = self.resnet(x)
        x1 = self.actor(x)
        x2 = self.critic(torch.cat([x,x1],dim=1))
        return x1,x2

    def training_step(self, batch, batch_idx):
        states, actions, returns = batch
        actions = actions.float()  # Convert actions tensor to Float type
        returns = returns.float()  # Convert returns tensor to Float type
        returns = torch.unsqueeze(returns, dim=-1)
        actor_output, critic_output = self.forward(states)

        # Compute actor and critic losses

        actor_loss = torch.nn.MSELoss()(actions.float(), actor_output.float())

        critic_loss = torch.nn.MSELoss()(returns, critic_output)
        total_loss = actor_loss+critic_loss

        # Return the losses as a dictionary

        # # Log the losses
        self.log('loss', total_loss,prog_bar=True)
        self.log('actor_loss', actor_loss,prog_bar=True)
        self.log('critic_loss', critic_loss,prog_bar=True)
        # self.log('critic_loss', critic_loss)
        # self.log('total_loss', total_loss)
        self.log('lr',self.learning_rate)


        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, r = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return [optimizer], [scheduler]

