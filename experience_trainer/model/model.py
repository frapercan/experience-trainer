import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
from torchvision.models import resnet18
import torch.nn.functional as F

import torchvision.transforms.functional as F

from transformers import AutoTokenizer, AutoModelForVideoClassification
from transformers import VivitConfig, VivitModel
from transformers import AutoImageProcessor, TimesformerConfig, TimesformerModel, TimesformerForVideoClassification

import torchvision.models as models

from torchvision.models import vit_b_16
from transformers.time_series_utils import LambdaLayer

from experience_trainer.model.autoencoder_actions import ActionsAutoEncoder
from experience_trainer.model.autoencoder_rewards import RewardsAutoEncoder
from experience_trainer.model.video_autoencoder import VideoAutoEncoder


class ActorCriticModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(ActorCriticModel, self).__init__()
        self.learning_rate = learning_rate

        self.state_conv = nn.Sequential(
            nn.Conv2d(3, 6, 3),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 2, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 1, 3),
            nn.BatchNorm2d(1),
            nn.Flatten(),
        )
        self.critic = nn.Sequential(
            nn.Flatten(),  # Aplana el tensor a [11, 3*56*56]
            nn.Linear(52 * 52, 52),  # Reducir a [11, 5]
            nn.BatchNorm1d(52),
            nn.ReLU(),
            nn.Linear(52, 13),  # Reducir a [11, 5]
            nn.BatchNorm1d(13),
            nn.ReLU(),
            nn.Linear(13, 1),  # Reducir a [11, 5]
            nn.BatchNorm1d(1),
        )
        self.actor = nn.Sequential(
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

    def forward(self, backward_images):
        state = self.state_conv(backward_images.squeeze(2))
        critic_output = self.critic(state)
        actor_input = torch.cat([state, critic_output], dim=1)
        actor_output = self.actor(actor_input)

        return critic_output, actor_output

    def training_step(self, batch, batch_idx):
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = batch
        critic_output, actor_output = self.forward(backward_images)
        reward = normalized_reward_previous[:, -1] == normalized_reward_forward[:, 0]
        critic_loss = torch.nn.MSELoss()(reward.float(), critic_output.float().squeeze())
        actor_loss = torch.nn.MSELoss()(action_one_hot_forward[:, 0], actor_output.float())
        total_loss = critic_loss + actor_loss
        self.log('loss', total_loss, prog_bar=True)
        self.log('critic_loss', critic_loss, prog_bar=True)
        self.log('actor_loss', actor_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = batch
        critic_output, actor_output = self.forward(backward_images)
        reward = normalized_reward_previous[:, -1] == normalized_reward_forward[:, 0]
        critic_loss = torch.nn.MSELoss()(reward.float(), critic_output.float().squeeze())
        actor_loss = torch.nn.MSELoss()(action_one_hot_forward[:, 0], actor_output.float())
        total_loss = critic_loss + actor_loss
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_critic_loss', critic_loss, prog_bar=True)
        self.log('val_actor_loss', actor_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return [optimizer], [scheduler]


class ActorCritic(pl.LightningModule):
    def __init__(self, num_actions, learning_rate=0.001):
        super(ActorCritic, self).__init__()
        self.learning_rate = learning_rate
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.actor = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Linear(517, 128),
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
        print(x.shape)
        x = self.features(x).squeeze()
        print(x.shape)
        x1 = self.actor(x.squeeze())
        print(x1.shape)
        x2 = self.critic(torch.cat([x, x1], dim=1))
        print(x2.shape)
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

        return total_loss

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

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return [optimizer], [scheduler]


def imshow(tensor):
    # Convertir tensor de CHW a HWC
    img = tensor.cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()