#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:48:27 2025

Text tokenization, padding goes here


@author: andrey
"""

from collections import Counter


# Normalize text: Sort properties alphabetically
def normalize_text(text):
    properties = [prop.strip() for prop in text.split(",")]
    sorted_properties = ", ".join(sorted(properties))  # Alphabetical order
    return sorted_properties

# Tokenize text: Split into tokens (whitespace-based)
def tokenize(text):
    return text.lower().split()

# Build vocabulary from the normalized and tokenized corpus
def build_vocab(corpus):
    tokenized_corpus = [tokenize(text) for text in corpus]
    token_counts = Counter(token for tokens in tokenized_corpus for token in tokens)
    sorted_tokens = sorted(token_counts.keys())  # Ensure consistent ordering
    vocab = {token: idx + 1 for idx, token in enumerate(sorted_tokens)}  # Reserve 0 for padding
    return vocab

# Convert text to token indices using the vocabulary
def text_to_indices(text, vocab):
    tokens = tokenize(text)
    return [vocab[token] for token in tokens if token in vocab]


# Padding and truncating function
def pad_or_truncate(sequence, max_len, pad_value=0):
    """Pad or truncate a sequence to a fixed length."""
    if len(sequence) < max_len:
        # Pad with the specified pad_value
        return sequence + [pad_value] * (max_len - len(sequence))
    else:
        # Truncate to the maximum length
        return sequence[:max_len]

# Update list_to_dict to include padding/truncation
def list_to_dict(text_list, vocab=None, max_len=10, pad_value=0):
    normalized_texts = [normalize_text(text) for text in text_list]


    if vocab is None:
        vocab = build_vocab(normalized_texts)  # Build vocabulary if not provided

    text_descriptions = {
        f"{i + 1}.jpg": pad_or_truncate(text_to_indices(normalized_texts[i], vocab), max_len, pad_value)
        for i in range(len(text_list))
    }
    return text_descriptions, vocab

# Padding function to make all text sequences equal in length
def pad_texts(texts, max_len):
    padded_texts = []
    for text in texts:
        padded_text = text + [0] * (max_len - len(text))  # Pad with zeros
        padded_texts.append(padded_text)
    return padded_texts

