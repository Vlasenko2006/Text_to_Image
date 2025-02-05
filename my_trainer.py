#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:54:55 2025

@author: andrey
"""

import torch
from torchvision import transforms
from tqdm import tqdm
from list_to_dict import pad_texts



def my_trainer(model,
               optimizer,
               criterion,
               train_loader,
               val_loader,
               device="cpu",
               num_epochs = 1
        ):   
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for texts, images in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            # Simplistic text preprocessing: map each character to its ASCII value
            text_sequences = [[ord(char) for char in text] for text in texts]
            max_len = max(len(seq) for seq in text_sequences)
            text_sequences = pad_texts(text_sequences, max_len)
            text_sequences = torch.tensor(text_sequences, dtype=torch.long).to(device)
    
            optimizer.zero_grad()
    
            outputs = model(text_sequences)
            loss = criterion(outputs, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
            optimizer.step()
    
            train_loss += loss.item()
    
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss / len(train_loader):.4f}")
    
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for texts, images in train_loader:
                images = images.to(device)
                text_sequences = [[ord(char) for char in text] for text in texts]
                max_len = max(len(seq) for seq in text_sequences)
                text_sequences = pad_texts(text_sequences, max_len)
                text_sequences = torch.tensor(text_sequences, dtype=torch.long).to(device)
    
                outputs = model(text_sequences)
                loss = criterion(outputs, images)
    
                val_loss += loss.item()
    
                # Save one output image per epoch
                if epoch == num_epochs - 1:  # Save in the final epoch
                    for i, output in enumerate(outputs):
                        output_img = transforms.ToPILImage()(output.cpu().detach().clamp(0, 1))
                        output_img.save(f"results_short/output_image_tr_{i}.png")
    
            
            
            
            for texts, images in val_loader:
                images = images.to(device)
                text_sequences = [[ord(char) for char in text] for text in texts]
                max_len = max(len(seq) for seq in text_sequences)
                text_sequences = pad_texts(text_sequences, max_len)
                text_sequences = torch.tensor(text_sequences, dtype=torch.long).to(device)
    
                outputs = model(text_sequences)
                loss = criterion(outputs, images)
    
                val_loss += loss.item()
    
                # Save one output image per epoch
                if epoch == num_epochs - 1:  # Save in the final epoch
                    for i, output in enumerate(outputs):
                        output_img = transforms.ToPILImage()(output.cpu().detach().clamp(0, 1))
                        output_img.save(f"results_short/output_image_V_{i}.png")
    
     #   print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.4f}")
    
    # Save the model
    torch.save(model.state_dict(), "results_short/text_to_image_model.pth")
    print("Training complete and results saved!")

