import torch
import numpy as np
import pandas as pd
import clip
from tqdm import tqdm
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the dataset (CSV with image_id and captions)
dataset = pd.read_csv("eng_dataset.csv")

# Check total number of rows in the CSV
print(f"Total number of rows in CSV: {len(dataset)}")

# Count the number of unique image_ids
unique_image_ids = dataset['image_id'].unique()
print(f"Number of unique image_ids: {len(unique_image_ids)}")

# Check for duplicate rows based on image_id
duplicate_rows = dataset.duplicated(subset=['image_id']).sum()
print(f"Number of duplicate rows: {duplicate_rows}")

# Check how many captions exist per image_id
captions_per_image = dataset.groupby('image_id').size()
print(f"Captions per image_id (example):\n{captions_per_image.head()}")

# Initialize lists for storing embeddings   
image_embeddings = []
image_ids = []

# Process each image
for image_id in tqdm(unique_image_ids, desc="Processing Images"):
    # Get all captions for the current image_id
    captions = dataset[dataset['image_id'] == image_id]['caption'].tolist()
    
    # Tokenize the captions
    text = clip.tokenize(captions, truncate=True).to(device)
    
    # Process the embeddings for the text (captions)
    with torch.no_grad():
        text_features = model.encode_text(text)
        
    # Normalize the text embeddings
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Append the embeddings and corresponding image_id
    image_embeddings.append(text_features.cpu().numpy())
    image_ids.append(image_id)

# Save embeddings to NPZ file
np.savez("caption_embeddings_clip.npz", image_ids=np.array(image_ids), embeddings=np.array(image_embeddings))

print("NPZ file saved: caption_embeddings_clip.npz")
