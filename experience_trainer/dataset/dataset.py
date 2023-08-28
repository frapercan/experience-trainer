import io
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchvision import transforms
from webdataset import WebDataset
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T

def bytes_to_tensor(byte_data):
    """Load a tensor from its binary representation."""
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer)


def calculate_reward(scores):
    if len(scores) == 1:
        return -1
    reward = []
    for i,score in enumerate(scores[:-1]):
        if scores[i+1] == score:
            reward.append(-1)
        else:
            reward.append(1)
    result = sum(reward)/(len(scores)-1)
    return result


class CustomIterableDataset(IterableDataset):
    def __init__(self,dataset_path,actions_mapping,dataset_length):
        self.dataset_path = dataset_path
        self.actions_mapping = actions_mapping
        self.initialize_keys()
        self.dataset_length = dataset_length



    def initialize_keys(self):
        self.dataset = WebDataset(self.dataset_path)


    def __iter__(self):
        for sample in self.dataset:
            rolling = len([key for key in sample.keys() if 'backward' in key])

            backward_images = [sample[f"image_backward_{i}.png"] for i in range(rolling)]
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            image = Image.open(io.BytesIO(backward_images[-1]))
            image_tensor = transform(image)
            state = json.loads(sample['json'].decode("utf-8"))

            action = state['rolling_5_action_ahead'][0]
            action_index = self.get_action_index(action)
            action_tensor = torch.tensor(action_index)
            reward = calculate_reward(state['rolling_5_score_ahead'])
            

            # Apply One-Hot Encoding to action
            action_one_hot = F.one_hot(action_tensor, num_classes=len(self.actions_mapping))
            yield image_tensor , action_one_hot , torch.tensor(reward).type(torch.FloatTensor)

    def get_action_index(self, action):
        # Define your logic to map action strings to indices
        # For example, if you have a list of action strings ['action1', 'action2', 'action3']
        # and you want to map them to indices [0, 1, 2], you can use the following code:
        return self.actions_mapping.index(action)

    def __len__(self):
        return self.dataset_length