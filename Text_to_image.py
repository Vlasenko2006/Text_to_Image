import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from list_to_dict import list_to_dict
from TextImageDataset import TextImageDataset
from TextToImageModel import TextToImageModel
from torch.utils.data import DataLoader
from my_trainer import my_trainer

# Preprocessing and loading data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Training loop
num_epochs = 29000
lr = 0.0000251

# Dummy dataset paths and descriptions
image_dir = "your path goes here"
text_list = ["your description goes here"]
val_text_list = ["your description goes here"]

text_descriptions, vocab = list_to_dict(text_list)
val_descriptions = list_to_dict(val_text_list)

# Dataset and dataloaders
dataset = TextImageDataset(image_dir, text_descriptions, transform=transform)
dataset_val = TextImageDataset(image_dir, val_descriptions, transform=transform)

train_dataset = dataset
val_dataset = dataset

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
val_loader2 = DataLoader(dataset_val, batch_size=8, shuffle=False)

# Initialize the model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextToImageModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Directory to save results
os.makedirs("results_short", exist_ok=True)

my_trainer(model,
           optimizer,
           criterion,
           train_loader,
           val_loader,
           device=device,
           num_epochs=1)

print("Training complete!")
