import logging
import os
import re

import torch
import yaml
import json
import pickle
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from experience_trainer.dataset.dataset import ExperienceIterableDataset
from experience_trainer.model.autoencoder_actions import ActionsAutoEncoder
from experience_trainer.model.autoencoder_rewards import RewardsAutoEncoder
from experience_trainer.model.model import ActorCriticModel
from experience_trainer.model.video_autoencoder import VideoAutoEncoder

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperienceTrainer:
    def __init__(self, config):
        logger.info("Initializing AutoEncoderHandler...")
        self.config = config
        self.dataset, self.val_dataset = self.load_datasets()
        self.lr_monitor = LearningRateMonitor(logging_interval='step')
        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'],
                                     num_workers=self.config['num_workers'])
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config['val_batch_size'],
                                         num_workers=self.config['num_workers'])

    def log_graph_to_tensorboard(self, model):

        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous, normalized_reward_forward = sample
        inputs = (
        backward_images, forward_images, action_one_hot_previous, action_one_hot_forward, normalized_reward_previous,
        normalized_reward_forward)

        tb_writer = SummaryWriter(log_dir=os.path.join(self.config['log_dir'], model.__class__.__name__))
        tb_writer.add_graph(model, inputs)

    def load_datasets(self):
        logger.info("Loading datasets...")
        with open(self.config['dataset_metadata'], "r") as metadata_file:
            metadata = json.load(metadata_file)

        with open(self.config['val_dataset_metadata'], "r") as metadata_file:
            val_metadata = json.load(metadata_file)

        dataset = ExperienceIterableDataset(self.config['dataset_path'], self.config['actions_mapping'],
                                            metadata['length'])
        val_dataset = ExperienceIterableDataset(self.config['val_dataset_path'], self.config['actions_mapping'],
                                                val_metadata['length'])

        with open(os.path.join(self.config['actions_map_path'], "map"), 'wb') as file:
            pickle.dump(dataset.actions_mapping, file)

        return dataset, val_dataset

    def setup_logger(self, model_class):
        logger.info(f"Setting up TensorBoard logger for {model_class.__name__}...")

        return TensorBoardLogger(save_dir=self.config['log_dir'], name=model_class.__name__, log_graph=True)

    def setup_checkpoint_callback(self, model_class):
        logger.info(f"Setting up checkpoint callback for {model_class.__name__}...")

        checkpoint_dir = os.path.join(self.config['model_save_path'], model_class.__name__)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        return ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=model_class.__name__,
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )

    def load_best_checkpoint(self, model_class):
        logger.info(f"Loading best checkpoint for {model_class.__name__}...")

        checkpoint_dir = os.path.join(self.config['model_save_path'], model_class.__name__)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]

        best_metric = float('inf')  # Lower is better (e.g., loss)
        best_checkpoint = None

        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

            # Load checkpoint without initializing the model
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

            # Extract the metric of interest
            metric_value = float(re.findall(r"tensor\((.*?)\)", str(checkpoint['callbacks']))[0])
            # Update the best metric and best checkpoint if necessary
            if metric_value is not None and metric_value < best_metric:
                best_metric = metric_value
                best_checkpoint = checkpoint_path

        if best_checkpoint:
            logger.info(f"Best checkpoint for {model_class.__name__} found at {best_checkpoint}")
        else:
            logger.warning(f"No checkpoint found for {model_class.__name__}")

        return best_checkpoint

    def train_model(self, model_class,epochs, *args, **kwargs):
        logger.info(f"Training model: {model_class.__name__}...")
        self.trainer = Trainer(
            max_epochs=epochs,
            callbacks=[self.setup_checkpoint_callback(model_class), self.lr_monitor],
            log_every_n_steps=5,
            logger=self.setup_logger(model_class)
        )
        model = model_class(learning_rate=self.config['lr'], *args, **kwargs)

        best_checkpoint = self.load_best_checkpoint(model_class)
        if best_checkpoint:
            checkpoint = torch.load(best_checkpoint)
            model.load_state_dict(checkpoint['state_dict'])

        if model_class == ActorCriticModel:
            checkpointVideoAutoEncoder =   torch.load(self.load_best_checkpoint(VideoAutoEncoder))
            checkpointActionsAutoEncoder = torch.load(self.load_best_checkpoint(ActionsAutoEncoder))
            checkpointRewardsAutoEncoder = torch.load(self.load_best_checkpoint(RewardsAutoEncoder))
            model.video_autoencoder.load_state_dict(checkpointVideoAutoEncoder['state_dict'])
            model.action_autoencoder.load_state_dict(checkpointActionsAutoEncoder['state_dict'])
            model.rewards_autoencoder.load_state_dict(checkpointRewardsAutoEncoder['state_dict'])

        self.trainer.fit(model, self.dataloader, self.val_dataloader)
        # self.log_graph_to_tensorboard(model)



if __name__ == "__main__":
    with open("config.yaml", 'r', encoding='utf-8') as conf:
        config = yaml.safe_load(conf)
    handler = ExperienceTrainer(config)
    handler.train_model(VideoAutoEncoder,epochs=config['epochs_video'], mode="encode")
    handler.train_model(ActionsAutoEncoder,epochs=config['epochs_actions'], mode="encode")
    handler.train_model(RewardsAutoEncoder,epochs=config['epochs_rewards'], mode="encode")

    handler.train_model(ActorCriticModel,epochs=config['epochs_actor_critic'])
