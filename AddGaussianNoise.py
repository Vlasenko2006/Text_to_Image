#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:52:18 2025

Adds Gaussian noise to images to avoid overfitting.

@author: andrey
"""

import torch.randn

class AddGaussianNoise:
    """Custom transform to add Gaussian noise to an image."""
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
