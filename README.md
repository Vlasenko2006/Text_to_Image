# Text-to-Image Model

This repository contains a PyTorch-based deep learning model that generates images from textual descriptions. The model leverages a combination of **LSTM-based text encoding, multi-head self-attention, and convolutional upsampling** to generate images from input text sequences.

## Why Combine LSTM with Self-Attention?  

Combining **LSTM** with **self-attention** allows the model to leverage the **strengths of both architectures** while compensating for their weaknesses.  

### 1. LSTM Captures Sequential Patterns (Short-Term Context)  
- LSTMs process text **sequentially**, which means they are good at capturing **local dependencies** and **order-sensitive** information.  
- They handle **long-term dependencies better than vanilla RNNs**, but they **still struggle with very long sequences** due to memory constraints.  

### 2. Self-Attention Captures Global Context Efficiently  
- **Self-attention** (like in Transformers) allows the model to **look at the entire sequence at once** instead of processing it step by step.  
- This helps capture **long-range dependencies** and relationships between words that might be far apart.  
- Unlike LSTMs, self-attention does not suffer from **vanishing gradients** over long sequences.  

### 3. Hybrid Approach = Best of Both Worlds  
By combining an **LSTM with self-attention**, we get:  
✅ **LSTM for local context & sequential understanding**  
✅ **Self-attention for capturing global dependencies**  

### 4. How it Works in this Model  
- The LSTM **processes the input sequentially**, learning relationships between neighboring tokens.  
- Then, the **self-attention mechanism** (via `MultiHeadSelfAttention`) helps refine these representations by looking at all words at once and adjusting their importance.  
- This makes the **text encoding more informative** before it is mapped to image features.  


---

## Features
- **Text Normalization & Tokenization**: Preprocesses text inputs by normalizing and tokenizing.
- **Custom Dataset Class**: Handles loading image-text pairs.
- **Data Augmentation**: Includes color jittering and Gaussian noise.
- **Residual Blocks & Self-Attention**: Enhances image generation quality.
- **Training & Validation Pipeline**: Supports training with loss tracking.

## Installation
To set up the required dependencies, use the provided Anaconda environment:

### Using Conda:
Save the following `.yaml` file as `environment.yaml` and install it using Anaconda.

```yaml
name: text_to_image_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pytorch
  - torchvision
  - torchaudio
  - tqdm
  - pillow
  - pip
  - pip:
      - numpy
      - matplotlib
```

Install the environment using:
```bash
conda env create -f environment.yaml
conda activate text_to_image_env
```

### Using pip:
Alternatively, install dependencies using pip:
```bash
pip install torch torchvision torchaudio tqdm pillow numpy matplotlib
```

## Usage
### 1. Prepare Dataset
Store images in a specified directory and provide corresponding textual descriptions.

### 2. Train the Model
Run the training script:
```bash
python train.py
```

### 3. Generate Images
After training, generate images from new text inputs by running:
```bash
python generate.py --input "your text here"
```

## Model Architecture
The model follows a **Text-to-Image Pipeline**:
1. **Text Embedding**: Tokenized text is passed through an LSTM.
2. **Multi-Head Self-Attention**: Captures textual dependencies.
3. **Fully Connected Layer**: Maps text features to image features.
4. **Convolutional Decoder**: Upsamples features to reconstruct an image.

## Results
The trained model outputs generated images based on input text. Example results can be found in the `results_short/` directory.

## Saving & Loading Model
To save the trained model:
```bash
python save_model.py
```
To load and use the trained model:
```python
import torch
from model import TextToImageModel

model = TextToImageModel()
model.load_state_dict(torch.load("results_short/text_to_image_model.pth"))
model.eval()
```

## License
This project is open-source under the MIT License.

---
### Contributions & Issues
Feel free to open an issue or a pull request for improvements!

