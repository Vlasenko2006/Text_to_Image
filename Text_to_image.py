#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:17:32 2025

@author: andrey
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



# Convert list to dictionary
def list_to_dict(text_list):
    text_descriptions = {
        f"{i + 1}.jpg": np.asarray(text) 
        for i, text in enumerate(text_list)
    }
    return text_descriptions



# Step 1: Tokenize descriptions into expressions
def tokenize_expressions(text_list):
    """
    Tokenize descriptions into expressions separated by commas.
    Args:
        text_list (list of str): List of text descriptions.
    Returns:
        list of list of str: Tokenized expressions for each description.
    """
    tokenized = [[expr.strip().lower() for expr in text.split(",")] for text in text_list]
    return tokenized


# Step 2: Build vocabulary with tolerance for spelling errors
def build_expression_vocab(text_list, threshold=0.8):
    """
    Build a vocabulary of expressions with similarity-based merging.
    Args:
        text_list (list of str): List of text descriptions.
        threshold (float): Similarity threshold to merge expressions.
    Returns:
        dict: Mapping of expressions to indices.
        dict: Mapping of expressions to canonical vocabulary terms.
    """
    tokenized = tokenize_expressions(text_list)
    all_expressions = list(set(expr for desc in tokenized for expr in desc))
    
    # Calculate cosine similarity between all pairs of expressions
    expression_vectors = np.array([np.array([ord(c) for c in expr]).mean(axis=0) for expr in all_expressions])
    similarity_matrix = cosine_similarity(expression_vectors.reshape(-1, 1))
    
    # Merge similar expressions
    merged_vocab = {}
    for i, expr in enumerate(all_expressions):
        for j, other_expr in enumerate(all_expressions):
            if i != j and similarity_matrix[i, j] >= threshold:
                merged_vocab[other_expr] = expr
            else:
                merged_vocab.setdefault(expr, expr)
    
    # Build vocabulary with unique canonical terms
    unique_vocab = sorted(set(merged_vocab.values()))
    vocab = {expr: idx for idx, expr in enumerate(unique_vocab)}
    return vocab, merged_vocab


def remove_duplicates(lst):
    """
    Removes duplicates from each sublist in a list of lists.
    """
    def deduplicate(sublist):
        seen = set()
        return [x for x in sublist if not (x in seen or seen.add(x))]
    
    return [deduplicate(sublist) for sublist in lst]



# Step 3: Encode descriptions into sequences of indices with padding
def encode_descriptions_with_padding(text_list, vocab, max_length=None, pad_token="<PAD>"):
    """
    Encode text descriptions into numerical sequences with padding.
    Args:
        text_list (list of str): List of text descriptions.
        vocab (dict): Vocabulary mapping expressions to indices.
        max_length (int): Maximum sequence length. If None, use the max length in the dataset.
        pad_token (str): Padding token to use.
    Returns:
        list of list of int: Encoded and padded descriptions.
        dict: Updated vocabulary including the pad token.
    """
    # Add the padding token to the vocabulary if not present
    if pad_token not in vocab:
        vocab[pad_token] = len(vocab)

    # Tokenize and encode each description
    tokenized = tokenize_expressions(text_list)
    encoded_descriptions = remove_duplicates([[vocab.get(expr, vocab[pad_token]) for expr in desc] for desc in tokenized])

    # Determine the maximum length
    if max_length is None:
        max_length = max(len(desc) for desc in encoded_descriptions)

    # Add padding to ensure all descriptions are of the same length
    padded_descriptions = [
        desc + [vocab[pad_token]] * (max_length - len(desc)) for desc in encoded_descriptions
    ]

    padded_descriptions_sorted = [sorted(sublist) for sublist in padded_descriptions]
    return padded_descriptions_sorted, vocab



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

# Define the augmentation pipeline
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

def combine_unique_sentences(list1, list2):
    """
    Combines words from two lists into new sentences and ensures the result contains
    only unique sentences not present in the original lists.

    Args:
        list1 (list): The first list of sentences.
        list2 (list): The second list of sentences.

    Returns:
        list: A list of unique combined sentences.
    """
    # Create sets of existing sentences for faster lookups
    existing_sentences = set(list1) | set(list2)

    # Initialize a set for unique combined sentences
    unique_sentences = set()

    # Iterate through all pairwise combinations of sentences from list1 and list2
    for sentence1 in list1:
        for sentence2 in list2:
            # Combine sentences
            combined_sentence = f"{sentence1} {sentence2}"
            
            # Check for uniqueness
            if combined_sentence not in existing_sentences and combined_sentence not in unique_sentences:
                unique_sentences.add(combined_sentence)

    # Return the result as a list
    return list(unique_sentences)


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



# Generate Image from Text
def generate_image(model, text, device, output_path="output_image.png"):
    model.eval()
    with torch.no_grad():
        output = model(text)
        output_img = transforms.ToPILImage()(output.squeeze(0).clamp(-1, 1) * 0.5 + 0.5)
        output_img.save(output_path)
        plt.imshow(output_img)
        plt.show()
        print(f"Image saved to {output_path}")

# Load Trained Model

path_to_model = 'path to model' #"/home/andrey/Downloads/results_sort/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextToImageModel().to(device)
model.load_state_dict(torch.load( path_to_model + "text_to_image_model.pth", map_location=device))


text_list = ['insert your text descriptiom here']

sample = text_list #sample + [text_list[-5]]


# Build vocabulary
vocab, merged_vocab = build_expression_vocab(text_list, threshold=1.99)

# Encode descriptions with padding
padded_texts, updated_vocab = encode_descriptions_with_padding(sample, vocab)

counter = 0
for i, seq in enumerate(padded_texts):
    print("padded_texts[i] = ", padded_texts[i])
    print("seq = ", seq)
    seq = torch.tensor([seq], dtype=torch.long)
    #seq = seq[:,None]
    generate_image(model, seq, device, output_path= path_to_model + "test_im_"+str(counter)+".png")
    counter += 1 
