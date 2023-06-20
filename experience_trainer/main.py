import os
import pickle

import yaml
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from experience_trainer.dataset.dataset import CustomIterableDataset
from experience_trainer.model.model import ResNetDQN




if __name__ == "__main__":
    with open("config.yaml", 'r', encoding='utf-8') as conf:
        conf = yaml.safe_load(conf)
        dataset_path = conf['dataset_path']


        custom_dataset = CustomIterableDataset(dataset_path)
        with open(os.path.join(conf['actions_map_path'],"map"), 'wb') as file:
            pickle.dump(custom_dataset.actions_mapping, file)
        model = ResNetDQN(num_actions=len(custom_dataset.actions_mapping))

        trainer = pl.Trainer(max_epochs=conf['epochs'])
        trainer.fit(model, DataLoader(custom_dataset, batch_size=conf['batch_size'], num_workers=conf['num_workers']))
