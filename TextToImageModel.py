#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:07:41 2025

@author: andrey
"""


import torch.nn as nn



# Residual Block
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels, eps=1e-5),  # Add epsilon for stability
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channels, eps=1e-5),
        )

    def forward(self, x):
        return x + self.block(x)

# Multi-Head Self-Attention Block
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# Main Model
class TextToImageModel(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=512, img_size=64, norm_layer='instance', num_heads=8):
        super(TextToImageModel, self).__init__()

        # Text Encoder with Self-Attention
        self.text_encoder = nn.Sequential(
            nn.Embedding(1000, embedding_dim),
            nn.LSTM(embedding_dim, hidden_dim, num_layers=8, batch_first=True, dropout=0.35),
        )
        self.text_attention = MultiHeadSelfAttention(hidden_dim, num_heads)

        # Fully connected layer to map text features to image features
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 4 * 1024),
            nn.ReLU(),
            nn.Linear(4 * 1024, 8 * 8 * 128),
            nn.ReLU()
        )

        # Normalization layer
        norm = nn.InstanceNorm2d if norm_layer == 'instance' else nn.BatchNorm2d

        # Upsample Block without Attention
        def upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                norm(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualBlock(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                norm(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Image Decoder
        self.upsample_block1 = upsample_block(128, 128)
        self.upsample_block2 = upsample_block(128, 64)
        self.upsample_block3 = upsample_block(64, 32)

        # Final Block
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            norm(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Normalize to [-1, 1]
        )

    def forward(self, text):
        # Encode text
        text_embedding, _ = self.text_encoder(text)
        text_embedding = self.text_attention(text_embedding)
        text_features = text_embedding[:, -1, :]  # Take the last hidden state

        # Map text features to image features
        x = self.fc(text_features)
        x = x.view(-1, 128, 8, 8)

        # Decoder blocks
        x1 = self.upsample_block1(x)
        x2 = self.upsample_block2(x1)
        x3 = self.upsample_block3(x2)

        # Final block
        output = self.final_block(x3)
        return output
