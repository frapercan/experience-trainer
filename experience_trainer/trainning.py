import logging
import os
import re

import torch

torch.set_float32_matmul_precision('high')

import yaml
import json
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from experience_trainer.dataset.dataset import ExperienceIterableDataset

from pytorch_lightning.callbacks import EarlyStopping

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperienceTrainer:
    def __init__(self, config):
        logger.info("Initializing AutoEncoderHandler...")
        self.config = config
        self.metadata = None
        self.dataset, self.val_dataset = self.load_datasets()
        self.lr_monitor = LearningRateMonitor(logging_interval='step')
        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'],
                                     num_workers=self.config['num_workers'])
        if self.config['validation']:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.config['val_batch_size'],
                                             num_workers=self.config['num_workers'])

    def load_datasets(self):
        logger.info("Loading datasets...")
        with open(self.config['dataset_metadata'], "r") as metadata_file:
            metadata = json.load(metadata_file)
            self.metadata = metadata

        with open(self.config['val_dataset_metadata'], "r") as metadata_file:
            val_metadata = json.load(metadata_file)

        dataset = ExperienceIterableDataset(self.config['dataset_path'], metadata['actions'],
                                            metadata['length'],shuffle=self.config['shuffle'],shuffle_buffer_size=self.config['shuffle_buffer_size'])

        val_dataset = None
        if self.config['validation']:
            with open(self.config['val_dataset_metadata'], "r") as metadata_file:
                val_metadata = json.load(metadata_file)
            val_dataset = ExperienceIterableDataset(self.config['val_dataset_path'], metadata['actions'],
                                                    val_metadata['length'])

        return dataset, val_dataset

    def setup_logger(self, model_class):
        logger.info(f"Setting up TensorBoard logger for {model_class.__name__}...")

        return TensorBoardLogger(save_dir=self.config['log_dir'], name=model_class.__name__, log_graph=True)

    def setup_checkpoint_callback(self, model_class):
        logger.info(f"Setting up checkpoint callback for {model_class.__name__}...")

        checkpoint_dir = os.path.join(self.config['checkpoints_path'], model_class.__name__)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        return ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=model_class.__name__,
            save_top_k=self.config['save_top_k'],
            monitor=self.config['checkpoint_metric'],
            mode=self.config['checkpoint_mode']
        )

    def setup_early_stopping_callback(self):
        early_stop_callback = EarlyStopping(
            monitor=self.config['early_stopping_metric'],
            min_delta=0.00,
            patience=self.config['early_stopping_patience'],
            verbose=True,
            mode=self.config['early_stopping_mode']
        )

        return early_stop_callback

    def load_best_checkpoint(self, model_class):
        logger.info(f"Loading best checkpoint for {model_class.__name__}...")

        checkpoint_dir = os.path.join(self.config['checkpoints_path'], model_class.__name__)
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

    def train_model(self, model_class, epochs, *args, **kwargs):
        logger.info(f"Training model: {model_class.__name__}...")
        callbacks = [self.lr_monitor]
        if self.config['checkpoint_callback']:
            callbacks.append(self.setup_checkpoint_callback(model_class))
        if self.config['early_stopping_callback']:
            callbacks.append(self.setup_early_stopping_callback())

        print(callbacks)
        self.trainer = Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            log_every_n_steps=10,
            logger=self.setup_logger(model_class)
        )
        model = model_class(self.config, learning_rate=self.config['lr'], *args, **kwargs)

        if self.config['load_checkpoint']:
            best_checkpoint = self.load_best_checkpoint(model_class)
            if best_checkpoint:
                checkpoint = torch.load(best_checkpoint)
                model.load_state_dict(checkpoint['state_dict'])

        if self.config['validation']:
            self.trainer.fit(model, self.dataloader, self.val_dataloader)
        else:
            self.trainer.fit(model, self.dataloader)

        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                # Supongamos que quieres acceder a la lista de las rutas de los checkpoints
                checkpoint_paths = callback.best_k_models.keys()
                print(checkpoint_paths)
