import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from multilingual_clip import pt_multilingual_clip
import transformers

# Load M-CLIP Model and Tokenizer
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name).to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset = pd.read_csv("mon_dataset.csv")

# Initialize lists for storing embeddings
image_embeddings = []
image_ids = []

# Set max token length similar to CLIP (77 tokens)
MAX_TOKENS = 77  

# Process each image
for image_id in tqdm(dataset['image_id'].unique(), desc="Processing Images"):
    # Get all captions for the current image_id
    captions = dataset[dataset['image_id'] == image_id]['caption'].tolist()
    
    # Tokenize with truncation
    tokenized_inputs = tokenizer(
        captions, 
        padding=True, 
        truncation=True, 
        max_length=MAX_TOKENS,
        return_tensors="pt"
    ).to(device)

    # Compute embeddings using M-CLIP
    with torch.no_grad():
        text_features = model.forward(captions, tokenizer)
    
    # Normalize embeddings
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Append embeddings and image_id
    image_embeddings.append(text_features.cpu().numpy())
    image_ids.append(image_id)

# Save to NPZ
np.savez("caption_embeddings_mclip.npz", image_ids=np.array(image_ids), embeddings=np.array(image_embeddings, dtype=object))

print("NPZ file saved: caption_embeddings_mclip.npz")