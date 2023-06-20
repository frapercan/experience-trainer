import io
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from webdataset import WebDataset
from PIL import Image
import torch.nn.functional as F

class CustomIterableDataset(IterableDataset):
    def __init__(self, dataset_path, reward=False):
        self.dataset_path = dataset_path
        self.actions = []
        self.actions_mapping = []
        self.initialize_keys()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to the input size of ResNet
            transforms.ToTensor(),  # Transform the image to a tensor
        ])
        self.reward = reward

    def initialize_keys(self):
        dataset = WebDataset(self.dataset_path)
        for sample in dataset:
            state = json.loads(sample['json'].decode("utf-8"))
            self.actions.append(state['action'])
        self.length = len(self.actions)
        self.actions_mapping = list(set(self.actions))

    def __iter__(self):
        dataset = WebDataset(self.dataset_path)
        for sample in dataset:
            image = Image.open(io.BytesIO(sample["image"])).convert("RGB")
            image = self.transform(image)  # Apply the image transformation
            state = json.loads(sample['json'].decode("utf-8"))

            action = state['action']
            action_index = self.get_action_index(action)
            action_tensor = torch.tensor(action_index)
            # Apply One-Hot Encoding to action
            action_one_hot = F.one_hot(action_tensor, num_classes=len(self.actions_mapping))
            if not self.reward:
                yield image, action_one_hot
            reward = torch.tensor(int(state['score']))
            yield image, action_one_hot

    def get_action_index(self, action):
        # Define your logic to map action strings to indices
        # For example, if you have a list of action strings ['action1', 'action2', 'action3']
        # and you want to map them to indices [0, 1, 2], you can use the following code:
        return self.actions_mapping.index(action)

    def __len__(self):
        return self.length