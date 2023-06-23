import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pytorch_lightning.tuner import Tuner
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import LearningRateFinder
from experience_trainer.dataset.dataset import CustomIterableDataset
from experience_trainer.model.model import ResNet

if __name__ == "__main__":
    with open("config.yaml", 'r', encoding='utf-8') as conf:
        conf = yaml.safe_load(conf)
        dataset_path = conf['dataset_path']

        custom_dataset = CustomIterableDataset(dataset_path,reward=False)


        with open(os.path.join(conf['actions_map_path'], "map"), 'wb') as file:
            pickle.dump(custom_dataset.actions_mapping, file)

        model = ResNet(num_actions=len(custom_dataset.actions_mapping), learning_rate=conf['lr'])

        # Create TensorBoard logger
        logger = TensorBoardLogger(save_dir=conf['log_dir'], name='experiments')

        # Create a ModelCheckpoint callback to save the best model
        checkpoint_callback = ModelCheckpoint(
            dirpath=conf['model_save_path'],
            filename='best_model',
            save_top_k=1,
            monitor='total_loss',
            mode='min'
        )
        learning_rate_finder = LearningRateFinder(min_lr=0.0001,max_lr=0.1)
        print(learning_rate_finder)
        lr_monitor = LearningRateMonitor(logging_interval='step')



        trainer = pl.Trainer(
            max_epochs=conf['epochs'],
            logger=logger,  # Pass the TensorBoard logger to the trainer
            callbacks=[checkpoint_callback,lr_monitor],  # Pass the ModelCheckpoint callback to the trainer
            log_every_n_steps=30,  # Log metrics every 100 steps
        )

        dataloader = DataLoader(custom_dataset, batch_size=conf['batch_size'], num_workers=conf['num_workers'])
        tuner = Tuner(trainer)
        tuner.lr_find(model,dataloader)
        print(trainer.model.learning_rate)




        trainer.fit(
            model,
            dataloader
        )

        # Save the trained model
        model_path = os.path.join(conf['model_save_path'], "model.pt")
        torch.save(model.state_dict(), model_path)
        print("Model saved successfully.")

        for data in custom_dataset:
            # print(data)
            x,y,r = data
            print(trainer.model(torch.unsqueeze(x, dim=0)))
