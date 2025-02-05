#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:58:43 2025

Prepares a dataset with text-image samples. 

transform_with_augmentation - is the default data augmentation

@author: andrey
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise

transform_with_augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust colors
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AddGaussianNoise(mean=0.0, std=0.1)  # Add Gaussian noise
])


# Define the dataset
class TextImageDataset(Dataset):
    def __init__(self, image_dir, text_descriptions, transform=transform_with_augmentation):
        self.image_dir = image_dir
        self.text_descriptions = text_descriptions  # A dictionary mapping image filenames to descriptions
        self.image_filenames = list(text_descriptions.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        text = self.text_descriptions[image_filename]

        if self.transform:
            image = self.transform(image)

        return text, image
