import math
import time

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import IterableDataset
from webdataset import WebDataset
import io
import json
import torch.nn.functional as F
from PIL import Image
import torch


class ExperienceIterableDataset(IterableDataset):
    def __init__(self, dataset_path, actions_mapping, dataset_length, shuffle=True, shuffle_buffer_size=10000):
        self.dataset_path = dataset_path
        self.actions_mapping = actions_mapping
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.initialize_keys()
        self.dataset_length = dataset_length

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def initialize_keys(self):
        self.dataset = WebDataset(self.dataset_path)
        if self.shuffle:
            self.dataset = self.dataset.shuffle(self.shuffle_buffer_size)

    def __iter__(self):
        for sample in self.dataset:
            image = self.transform(Image.open(io.BytesIO(sample['image.png'])))
            state = json.loads(sample['json'].decode("utf-8"))
            action = F.one_hot(self.get_action_index(state['action']), num_classes=len(self.actions_mapping)).type(
                torch.FloatTensor)

            score_ahead = state['rolling_5_score_ahead']
            score_ahead = [float(score) if score != "None" else 0 for score in score_ahead]
            score_ahead = torch.tensor(score_ahead)

            reward = calculate_reward(score_ahead)
            yield image, action, reward

    def get_action_index(self, action):
        return torch.tensor(self.actions_mapping.index(action))

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


def decode_pt(data):
    """Decode a .pt tensor file from raw bytes."""
    stream = io.BytesIO(data)
    return torch.load(stream)


def bytes_to_tensor(byte_data):
    """Load a tensor from its binary representation."""
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer)


def calculate_reward(scores):
    differences = torch.abs(torch.diff(scores))

    return torch.tensor(float(1)) if differences[0] > 0 else torch.tensor(float(0))
