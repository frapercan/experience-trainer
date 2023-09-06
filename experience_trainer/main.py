import json
import os
import pickle

import matplotlib.pyplot as plt
import torch
import yaml
from experience_trainer.dataset.dataset import ExperienceDataset, ExperienceIterableDataset
from experience_trainer.model.autoencoder_actions import ActionsAutoEncoder
from experience_trainer.model.autoencoder_rewards import RewardsAutoEncoder
from experience_trainer.model.model import ActorCriticModel
from experience_trainer.model.video_autoencoder import VideoAutoEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    with open("config.yaml", 'r', encoding='utf-8') as conf:
        config = yaml.safe_load(conf)

        with open(config['dataset_metadata'], "r") as metadata_file:
            metadata = json.load(metadata_file)

        with open(config['val_dataset_metadata'], "r") as metadata_file:
            val_metadata = json.load(metadata_file)

        dataset_path = config['dataset_path']
        dataset = ExperienceIterableDataset(dataset_path, config['actions_mapping'], metadata['length'])

        val_dataset_path = config['val_dataset_path']
        val_dataset = ExperienceIterableDataset(val_dataset_path, config['actions_mapping'], val_metadata['length'])

        with open(os.path.join(config['actions_map_path'], "map"), 'wb') as file:
            pickle.dump(dataset.actions_mapping, file)

        # Logger for TensorBoard
        logger = TensorBoardLogger(save_dir=config['log_dir'], name='experiments', log_graph=True)

        # Checkpoint to save the best model
        checkpoint_callback = ModelCheckpoint(
            dirpath=config['model_save_path'],
            filename='best_model',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = Trainer(
            max_epochs=config['epochs'],
            callbacks=[checkpoint_callback, lr_monitor],
            log_every_n_steps=5,
        )

        dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
        val_dataloader = DataLoader(val_dataset, batch_size=config['val_batch_size'], num_workers=config['num_workers'])

        video_autoencoder_model = VideoAutoEncoder(learning_rate=config['lr'],mode="encode")
        trainer.fit(video_autoencoder_model, dataloader, val_dataloader)

        actions_autoencoder_model = ActionsAutoEncoder(learning_rate=config['lr'],mode="encode")
        trainer.fit(actions_autoencoder_model, dataloader, val_dataloader)

        rewards_autoencoder_model = RewardsAutoEncoder(learning_rate=config['lr'],mode="encode")
        trainer.fit(rewards_autoencoder_model, dataloader, val_dataloader)

        # model = ActorCriticModel()
        # trainer.fit(model, dataloader, val_dataloader)

        # Save the trained model
        # model_path = os.path.join(config['model_save_path'], "model-test.pt")
        # torch.save(ae_model.state_dict(), model_path)
        # print("Model saved successfully.")
        #
        # model = VideoAutoEncoder()
        # checkpoint = torch.load('/home/xaxi/PycharmProjects/experience-trainer/models/checkpoints/model-test.pt')
        # model.load_state_dict(checkpoint)
        # model.eval()

        # with torch.no_grad():
        #     for data in dataset:
        #         backward_images, forward_images, actions, rewards = data
        #         backward_images = backward_images.unsqueeze(0)
        #         actions = actions.unsqueeze(0)
        #         prediction = model([backward_images, forward_images, actions])
        #         video_prediction = prediction.permute((0, 2, 1, 3, 4))
        #
        #
        #         for frame in video_prediction[-1]:
        #             plt.imshow(frame.permute(1, 2, 0).detach().numpy())
        #             plt.show()
        #             break
