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


class VideoAutoEncoder(pl.LightningModule):
    def __init__(self, mode="encode", learning_rate=0.001):
        super(VideoAutoEncoder, self).__init__()
        self.mode = mode
        self.learning_rate = learning_rate

        # Encoder Bock
        self.conv1 = nn.Conv2d(3, 2, kernel_size=113)
        self.batch_norm1 = nn.BatchNorm2d(2)

        self.conv2 = nn.Conv2d(2, 1, kernel_size=57)
        self.batch_norm1 = nn.BatchNorm2d(1)

        self.latent = nn.Identity()

        self.relu = nn.ReLU()

        if mode == 'decode':
            self.freeze_encoder_weights(True)
        if mode == 'encode':
            self.freeze_encoder_weights(False)

    def freeze_encoder_weights(self, value):
        pass
        # for param in self.conv1.parameters():
        #     param.requires_grad = not value
        # for param in self.batch_norm1.parameters():
        #     param.requires_grad = not value
        # for param in self.conv2.parameters():
        #     param.requires_grad = not value
        # for param in self.batch_norm2.parameters():
        #     param.requires_grad = not value
        # for param in self.conv3.parameters():
        #     param.requires_grad = not value
        # for param in self.batch_norm3.parameters():
        #     param.requires_grad = not value
        # for param in self.conv4.parameters():
        #     param.requires_grad = not value
        # for param in self.batch_norm4.parameters():
        #     param.requires_grad = not value

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

            if self.mode == 'd':
                self.freeze_encoder()

            return optimizer

    def forward(self, x):
        x = x.squeeze()

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        latent = self.latent(x)



        #
        # if self.mode == 'encode':
        #     return x
        # if self.mode == 'decode':
        #     return latent


    def training_step(self, batch, batch_idx):
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = batch

        video_output = self.forward(backward_images)
        video_loss = torch.nn.MSELoss()(backward_images, video_output.float())

        self.log('loss', video_loss, prog_bar=True)
        self.log('lr', self.learning_rate)

        return video_loss

    def validation_step(self, batch, batch_idx):
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = batch

        video_output = self.forward(backward_images)
        video_loss = torch.nn.MSELoss()(backward_images, video_output.float())

        self.log('val_loss', video_loss, prog_bar=True)
        self.log('lr', self.learning_rate)

        return video_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

        return [optimizer], [scheduler]
