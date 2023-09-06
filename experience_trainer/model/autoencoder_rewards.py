import time

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


class RewardsAutoEncoder(pl.LightningModule):
    def __init__(self, mode="encode", learning_rate=0.001):
        super(RewardsAutoEncoder, self).__init__()
        self.mode = mode
        self.learning_rate = learning_rate

        # Encoder Bock
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=28, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(28)

        self.conv2 = nn.Conv1d(in_channels=28, out_channels=56, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(56)

        # Bottleneck
        self.conv_transpose = nn.ConvTranspose1d(in_channels=56, out_channels=56, kernel_size=52)
        self.batch_norm3 = nn.BatchNorm1d(56)
        self.latent = nn.Identity()

        # Decoder Block

        self.deconv1 = nn.ConvTranspose1d(in_channels=56, out_channels=28, kernel_size=1)
        self.batch_norm3 = nn.BatchNorm1d(28)

        self.deconv2 = nn.ConvTranspose1d(in_channels=28, out_channels=1, kernel_size=1)
        self.batch_norm4 = nn.BatchNorm1d(1)

        self.relu = nn.ReLU()

        if mode == 'decode':
            self.freeze_encoder_weights(True)
        if mode == 'encode':
            self.freeze_encoder_weights(False)

    def freeze_encoder_weights(self, value):
        for param in self.conv1.parameters():
            param.requires_grad = not value
        for param in self.batch_norm1.parameters():
            param.requires_grad = not value
        for param in self.conv2.parameters():
            param.requires_grad = not value
        for param in self.batch_norm2.parameters():
            param.requires_grad = not value

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        latent = self.conv_transpose(x)
        latent = self.batch_norm2(latent)
        latent = self.relu(latent).unsqueeze(1)

        x = self.deconv1(x)
        x = self.batch_norm3(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.batch_norm4(x)
        x = self.relu(x)

        x = x.squeeze(1)

        if self.mode == 'encode':
            return x
        if self.mode == 'decode':
            return latent

    def training_step(self, batch, batch_idx):
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = batch
        rewards_output = self.forward(normalized_reward_previous)
        rewards_loss = torch.nn.MSELoss()(rewards_output, normalized_reward_previous.float())

        self.log('loss', rewards_loss, prog_bar=True)
        self.log('lr', self.learning_rate)

        return rewards_loss

    def validation_step(self, batch, batch_idx):
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = batch

        rewards_output = self.forward(normalized_reward_previous)
        rewards_loss = torch.nn.MSELoss()(rewards_output, rewards_output.float())

        self.log('val_loss', rewards_loss, prog_bar=True)
        self.log('lr', self.learning_rate)

        return rewards_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return [optimizer], [scheduler]
