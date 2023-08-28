import json
import os
import pickle
import sys
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
from experience_trainer.model.model import ActorCritic

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

if __name__ == "__main__":


    with open("config.yaml", 'r', encoding='utf-8') as conf:
        conf = yaml.safe_load(conf)

        metadata_file =  open(conf['dataset_metadata'], "r")
        metadata = json.load(metadata_file)
        dataset_length = metadata['length']

        dataset_path = conf['dataset_path']

        actions_mapping = conf['actions_mapping']
        custom_dataset = CustomIterableDataset(dataset_path,actions_mapping,dataset_length)

        for sample in custom_dataset:
            pass
        with open(os.path.join(conf['actions_map_path'], "map"), 'wb') as file:
            pickle.dump(custom_dataset.actions_mapping, file)

        model = ActorCritic(num_actions=len(custom_dataset.actions_mapping), learning_rate=conf['lr'])

        # model.model
        # Create TensorBoard logger
        # logger = TensorBoardLogger(save_dir=conf['log_dir'], name='experiments',log_graph=True)
        # logger._log_graph = True
        # logger = TensorBoardLogger(save_dir=conf['log_dir'], name='experiments', log_graph=True)
        # logger._log_graph = True

        # Set the example_input_array attribute of the model
        # model.example_input_array = torch.ones((1, 3, 256, 256))

        # Alternatively, you can pass the input_array argument when calling log_graph
        # logger.log_graph(model, input_array=torch.ones((1, 3, 256, 256)))
        # Create a ModelCheckpoint callback to save the best model
        checkpoint_callback = ModelCheckpoint(
            dirpath=conf['model_save_path'],
            filename='best_model',
            save_top_k=1,
            monitor='loss',
            mode='min'
        )
        learning_rate_finder = LearningRateFinder(min_lr=0.00003,max_lr=0.03,num_training_steps=1000)
        # lr_monitor = LearningRateMonitor(logging_interval='step')



        trainer = pl.Trainer(
            max_epochs=conf['epochs'],
            # logger=logger,  # Pass the TensorBoard logger to the trainer
            callbacks=[checkpoint_callback],  # Pass the ModelCheckpoint callback to the trainer
            log_every_n_steps=5,  # Log metrics every 100 steps
        )

        dataloader = DataLoader(custom_dataset, batch_size=conf['batch_size'], num_workers=conf['num_workers'])

        # tuner.lr_find(model,dataloader)
        # print(trainer.model.learning_rate)




        trainer.fit(
            model,
            dataloader
        )


        # Save the trained model
        model_path = os.path.join(conf['model_save_path'], "model.pt")
        torch.save(model.state_dict(), model_path)
        print("Model saved successfully.")


        model = ActorCritic(num_actions=5)
        checkpoint = torch.load('/home/xaxi/PycharmProjects/experience-trainer/models/checkpoints/best_model-v70.ckpt')
        model.load_state_dict(checkpoint['state_dict'])

        i=0
        for data in custom_dataset:
            # print(data)
            x,y,r = data
            print(model(torch.unsqueeze(x, dim=0)))
            print(y,r)
            i+=1
            if i == 100:
                break
