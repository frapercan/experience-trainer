import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
from torchvision.models import densenet201, DenseNet201_Weights
from torchmetrics import Accuracy
from experience_trainer.focal import FocalLoss
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

from experience_trainer.models.cross_attention_fusion import CrossAttentionRegression


class ActorCriticEfficientNet(pl.LightningModule):
    def __init__(self, config, metadata, criterion=FocalLoss(), learning_rate=0.001):
        super(ActorCriticEfficientNet, self).__init__()
        self.config = config

        self.criterion = criterion
        self.learning_rate = learning_rate
        self.actions = metadata['actions']
        self.automatic_optimization = False
        self.example_input_array = torch.randn((1, 3, 224, 224))

        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.actions), top_k=1)
        self.accuracy_val = Accuracy(task="multiclass", num_classes=len(self.actions), top_k=1)

        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')

        # Obtener el número de características de la última capa convolucional
        in_features = self.backbone._fc.in_features

        # Reemplazar la capa FC original para ajustarse a tu número de clases
        self.backbone._fc = nn.Linear(in_features, len(self.actions))
        self.expand_action = nn.Linear(len(self.actions),in_features)
        self.cross_attention = nn.MultiheadAttention(embed_dim=in_features, num_heads=1)
        self.critic = nn.Linear(in_features,1)


    def forward(self, x):
        x = self.backbone.extract_features(x)
        # Nota: _swish es la activación de la última capa convolucional en EfficientNet
        conv_features = self.backbone._swish(x)
        # Pooling y clasificación
        x = self.backbone._avg_pooling(conv_features)
        x = x.flatten(start_dim=1)

        logits = self.backbone._fc(x)



        expanded_action = self.expand_action(logits)

        logits_state_features,weights = self.cross_attention(expanded_action,x,x)
        regression_value = self.critic(logits_state_features)

        # Retorna tanto las características como los logits
        return logits, conv_features,regression_value

    def training_step(self, batch, batch_idx):
        state, action, reward = batch
        # Actor + Feature Extractor
        logits, features, regression_value = self.forward(state)

        # Zero the gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Calculate losses
        actor_loss = self.criterion(logits, action)
        critic_loss = torch.nn.functional.mse_loss(regression_value.squeeze(1), reward)

        # Backward pass for actor and critic networks
        actor_loss.backward(retain_graph=True)  # retain_graph to allow critic_loss backward after this
        critic_loss.backward()

        # Step the optimizers
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        acc = self.accuracy(logits.argmax(dim=1), action.argmax(dim=1))

        total_loss = actor_loss + critic_loss

        self.log('actor_loss', actor_loss, prog_bar=True)
        self.log('critic_loss', critic_loss, prog_bar=True)
        self.log('total_loss', total_loss, prog_bar=True)

        self.log('train_acc_step', acc, prog_bar=True)



    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.accuracy, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        state, action, reward = batch
        # Actor + Feature Extractor
        logits, features, regression_value = self.forward(state)

        # Zero the gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Calculate losses
        val_actor_loss = self.criterion(logits, action)
        val_critic_loss = torch.nn.functional.mse_loss(regression_value.squeeze(1), reward)

        val_total_loss = val_actor_loss + val_critic_loss


        val_acc = self.accuracy_val(logits.argmax(dim=1), action.argmax(dim=1))


        self.log('val_actor_loss', val_actor_loss, prog_bar=True)
        self.log('val_critic_loss', val_critic_loss, prog_bar=True)
        self.log('val_total_loss', val_total_loss, prog_bar=True)

        self.log('train_acc_step', val_acc, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy_val, prog_bar=True)

    def configure_optimizers(self):
        # Separate the parameters of actor and critic
        actor_params = list(self.backbone.parameters()) + list(self.expand_action.parameters()) + list(self.cross_attention.parameters())
        critic_params = self.critic.parameters()

        # Create optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.learning_rate)

        # Check if learning rate scheduler should be used
        if self.config['CosineAnnealingLR']:
            actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, self.config['CosineAnnealingLR_steps'])
            critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, self.config['CosineAnnealingLR_steps'])
            return [self.actor_optimizer, self.critic_optimizer], [actor_scheduler, critic_scheduler]

        else:
            return [self.actor_optimizer, self.cr
                    itic_optimizer]


