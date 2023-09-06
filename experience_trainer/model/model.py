import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DiffusionPipeline
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

        self.video_autoencoder = VideoAutoEncoder(mode="decode")
        self.action_autoencoder = ActionsAutoEncoder(mode="decode")
        self.rewards_autoencoder = RewardsAutoEncoder(mode="decode")

        # self.latent_mlp = nn.Linear()

        self.critic = nn.Sequential(
            nn.Flatten(),  # Aplana el tensor a [11, 3*56*56]
            nn.Linear(2 * 56 * 56, 5),  # Reducir a [11, 5]
            nn.BatchNorm1d(5),
            nn.ReLU()
        )

        # Encoder Bock
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=28, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(28)

        self.conv2 = nn.Conv1d(in_channels=28, out_channels=56, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(56)

        # Bottleneck
        self.conv_transpose = nn.ConvTranspose1d(in_channels=56, out_channels=56, kernel_size=52)
        self.batch_norm3 = nn.BatchNorm1d(56)
        self.latent = nn.Identity()

        self.relu = nn.ReLU()

        self.actor = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.AvgPool2d(kernel_size=11, stride=11),
            nn.BatchNorm2d(num_features=1)
        )

    def forward(self, backward_images, action_one_hot_previous):
        video_latent = self.video_autoencoder(backward_images)
        actions_latent = self.action_autoencoder(action_one_hot_previous)

        critic_input = torch.cat([video_latent, actions_latent], dim=1)

        critic_output = self.critic(critic_input)


        actor_input_reward = self.conv1(critic_output.unsqueeze(1))
        actor_input_reward = self.batch_norm1(actor_input_reward)
        actor_input_reward = self.relu(actor_input_reward)

        actor_input_reward = self.conv2(actor_input_reward)
        actor_input_reward = self.batch_norm2(actor_input_reward)
        actor_input_reward = self.relu(actor_input_reward)

        actor_input_reward = self.conv_transpose(actor_input_reward)
        actor_input_reward = self.batch_norm2(actor_input_reward)
        actor_input_reward = self.relu(actor_input_reward).unsqueeze(1)

        actor_input = torch.cat([video_latent,actor_input_reward],dim=1)


        actor_output = self.actor(actor_input).squeeze()

        return critic_output,actor_output

    def training_step(self, batch, batch_idx):
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = batch
        critic_output,actor_output = self.forward(backward_images, forward_images, action_one_hot_previous, action_one_hot_forward,
                                     normalized_reward_previous, normalized_reward_forward)
        critic_loss = torch.nn.MSELoss()(normalized_reward_forward, critic_output.float())
        actor_loss = torch.nn.MSELoss()(action_one_hot_forward, actor_output.float())

        total_loss = critic_loss+actor_loss
        self.log('loss', total_loss, prog_bar=True)
        self.log('critic_loss', critic_loss, prog_bar=True)
        self.log('actor_loss', actor_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = batch
        critic_output,actor_output = self.forward(backward_images, forward_images, action_one_hot_previous, action_one_hot_forward,
                                     normalized_reward_previous, normalized_reward_forward)
        critic_loss = torch.nn.MSELoss()(normalized_reward_forward, critic_output.float())
        actor_loss = torch.nn.MSELoss()(action_one_hot_forward, actor_output.float())

        total_loss = critic_loss+actor_loss
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_critic_loss', critic_loss, prog_bar=True)
        self.log('val_actor_loss', actor_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return [optimizer], [scheduler]
