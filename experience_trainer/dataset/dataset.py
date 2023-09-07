import io
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from webdataset import WebDataset
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T


def decode_pt(data):
    """Decode a .pt tensor file from raw bytes."""
    stream = io.BytesIO(data)
    return torch.load(stream)


def bytes_to_tensor(byte_data):
    """Load a tensor from its binary representation."""
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer)


def calculate_reward(scores):
    scores = torch.tensor([float(score) if score != 'None' else 0 for score in scores])
    scores = torch.diff(scores)
    min_val = scores.min()
    max_val = scores.max()
    scores = (scores - min_val) / (max_val - min_val)

    weights = torch.linspace(len(scores), 1, len(scores))

    reward = torch.sum(scores * weights) / torch.sum(weights)






    return reward


class ExperienceIterableDataset(IterableDataset):
    def __init__(self, dataset_path, actions_mapping, dataset_length):
        self.dataset_path = dataset_path
        self.actions_mapping = actions_mapping
        self.initialize_keys()
        self.dataset_length = dataset_length

    def initialize_keys(self):
        self.dataset = WebDataset(self.dataset_path)

    def __iter__(self):
        for sample in self.dataset:
            rolling = len([key for key in sample.keys() if 'backward' in key])
            backward_image_bytes = sample[f'image_backward_0.pt']

            backward_images = torch.stack(
                [decode_pt(sample[f"image_backward_{i}.pt"]) for i in range(rolling)]).permute([1, 0, 2, 3])
            forward_images = torch.stack([decode_pt(sample[f"image_onward_{i}.pt"]) for i in range(rolling)]).permute(
                [1, 0, 2, 3])

            state = json.loads(sample['json'].decode("utf-8"))

            actions_previous = state['rolling_5_action_previous']
            actions_previous_index = [self.get_action_index(action) for action in actions_previous]
            actions_previous_tensor = torch.tensor(actions_previous_index)

            actions_forward = state['rolling_5_action_ahead']
            actions_forward_index = [self.get_action_index(action) for action in actions_forward]
            actions_forward_tensor = torch.tensor(actions_forward_index)

            # Apply One-Hot Encoding to action
            action_one_hot_previous = F.one_hot(actions_previous_tensor, num_classes=len(self.actions_mapping)).type(
                torch.FloatTensor)
            action_one_hot_forward = F.one_hot(actions_forward_tensor, num_classes=len(self.actions_mapping)).type(
                torch.FloatTensor)

            reward_previous = [float(score) if score != 'None' else 0 for score in state['rolling_5_score_previous']]
            reward_previous = torch.tensor(reward_previous)


            reward_forward = [float(score) if score != 'None' else 0 for score in state['rolling_5_score_ahead']]
            reward_forward = torch.tensor(reward_forward)

            reward = calculate_reward([state['rolling_5_score_previous'][-1]]+state['rolling_5_score_ahead'])
            yield backward_images.squeeze(), action_one_hot_forward[0], reward

    def get_action_index(self, action):
        return self.actions_mapping.index(action)

    def __len__(self):
        return self.dataset_length


class ExperienceDataset(Dataset):
    def __init__(self, dataset_path, actions_mapping, dataset_length):
        self.dataset_path = dataset_path
        self.actions_mapping = actions_mapping
        self.initialize_keys()
        self.dataset_length = dataset_length

    def initialize_keys(self):
        self.dataset = list(WebDataset(self.dataset_path))

    def __getitem__(self, index):
        sample = self.dataset[index]
        rolling = len([key for key in sample.keys() if 'backward' in key])
        backward_images = torch.stack([decode_pt(sample[f"image_backward_{i}.pt"]) for i in range(rolling)]).permute(
            [1, 0, 2, 3])
        forward_images = torch.stack([decode_pt(sample[f"image_onward_{i}.pt"]) for i in range(rolling)]).permute(
            [0, 1, 2, 3])
        state = json.loads(sample['json'].decode("utf-8"))
        action = state['rolling_5_action_ahead'][0]
        action_index = self.get_action_index(action)
        action_tensor = torch.tensor(action_index)
        reward = calculate_reward(state['rolling_5_score_ahead'])
        # Apply One-Hot Encoding to action
        action_one_hot = F.one_hot(action_tensor, num_classes=len(self.actions_mapping))
        return backward_images, forward_images, action_one_hot, torch.tensor(reward).type(torch.FloatTensor)

    def get_action_index(self, action):
        return self.actions_mapping.index(action)

    def __len__(self):
        return self.dataset_length
