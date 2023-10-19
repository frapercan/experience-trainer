import math

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torchmetrics import Accuracy
from torchvision.models import resnet18
from experience_trainer.focal import FocalLoss
import time

class ActorCriticResnet(pl.LightningModule):
    def __init__(self, config, metadata, criterion=FocalLoss(), learning_rate=0.001):
        super(ActorCriticResnet, self).__init__()
        self.config = config


        self.criterion = criterion
        self.learning_rate = learning_rate
        self.actions = metadata['actions']

        self.example_input_array = torch.randn((1, 3, 224, 224))

        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.actions), top_k=1)
        self.accuracy_val = Accuracy(task="multiclass", num_classes=len(self.actions), top_k=1)

        # State embedding layers
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.actor = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, len(self.actions)),
            nn.BatchNorm1d(len(self.actions)),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Linear(516, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x).squeeze([2,3])
        x1 = self.actor(x)
        x2 = self.critic(torch.cat([x, x1], dim=1))
        return x1, x2

    def training_step(self, batch, batch_idx):
        state, action, reward = batch

        actor_output, critic_output = self.forward(state)

        critic_output = critic_output.squeeze()
        # Compute actor and critic losses

        actor_loss = torch.nn.MSELoss()(action, actor_output)

        critic_loss = torch.nn.MSELoss()(reward, critic_output)
        total_loss = actor_loss + critic_loss

        # # Log the losses
        self.log('loss', total_loss, prog_bar=True)
        self.log('actor_loss', actor_loss, prog_bar=True)
        self.log('critic_loss', critic_loss, prog_bar=True)

        acc = self.accuracy(actor_output.argmax(dim=1), action.argmax(dim=1))
        self.log('train_acc_step', acc, prog_bar=True)

        return total_loss

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.accuracy, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        state, action, reward = batch

        actor_output, critic_output = self.forward(state)

        critic_output = critic_output.squeeze()
        # Compute actor and critic losses

        actor_loss = torch.nn.MSELoss()(action, actor_output)
        critic_loss = torch.nn.MSELoss()(reward, critic_output)

        total_loss = actor_loss + critic_loss

        # # Log the losses
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_actor_loss', actor_loss, prog_bar=True)
        self.log('val_critic_loss', critic_loss, prog_bar=True)

        val_acc = self.accuracy_val(actor_output.argmax(dim=1), action.argmax(dim=1))


        self.log('val_acc_step', val_acc, prog_bar=True)

        return total_loss


    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy_val, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.config['CosineAnnealingLR']:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config['CosineAnnealingLR_steps'])
            return [optimizer], [scheduler]

        else:
            return optimizer
