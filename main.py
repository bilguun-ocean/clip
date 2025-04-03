import os
import torch
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#  File paths
csv_file = "eng_captions.csv"
npz_file = "text_embeddings.npz"

#  Batch size
BATCH_SIZE = 32  

if os.path.exists(csv_file):
    print(f"{csv_file} found. Processing captions and saving embeddings...")

    #  Read CSV
    df = pd.read_csv(csv_file)

    #  Extract file_name and captions
    file_names = df['file_name'].to_numpy()  # Keep track of corresponding file names
    captions = df['caption'].astype(str).to_numpy()  # Ensure captions are strings

    #  Store embeddings
    text_embeddings = []

    print("Generating text embeddings...")

    #  Process captions in batches
    for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Processing Captions"):
        batch_captions = captions[i:i + BATCH_SIZE].tolist()  # Convert NumPy array slice to list
        inputs = processor(text=batch_captions, return_tensors="pt", padding=True, truncation=True).to(device)

        #  Get text embeddings
        with torch.no_grad():
            batch_embeddings = model.get_text_features(inputs["input_ids"])

        text_embeddings.append(batch_embeddings.cpu().numpy())  # Move to CPU and store

    #  Convert list of embeddings to a single NumPy array
    text_embeddings = np.vstack(text_embeddings)  # Shape: (num_samples, 512)

    #  Save file names and embeddings as NPZ
    np.savez(npz_file, file_name=file_names, caption=text_embeddings)
    print(f"Saved embeddings to {npz_file}")

else:
    print(f"{csv_file} not found. Please provide a valid CSV file.")
