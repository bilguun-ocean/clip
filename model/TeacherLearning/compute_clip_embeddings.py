import clip
import torch
from tqdm import tqdm
import pandas as pd

# Load the pre-trained CLIP model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device)

# Load your custom dataset (CSV with English captions and corresponding image IDs)
df = pd.read_csv('dataset.csv')

# Function to compute text embeddings
def compute_clip_embeddings(texts):
    # Load the CLIP model
    model, preprocess = clip.load("ViT-B/32", device)

    embeddings = []

    # Create a tqdm progress bar for the texts
    for text in tqdm(texts, desc="Processing captions", unit="caption"):
        # Tokenize the text with truncation enabled
        text_input = clip.tokenize([text], truncate=True).to(device)

        # Get the embeddings for the text
        with torch.no_grad():
            text_features = model.encode_text(text_input)
        
        embeddings.append(text_features.cpu().numpy())

    return embeddings

# Compute embeddings for English captions
english_embeddings = compute_clip_embeddings(df['eng'].tolist())

# Save embeddings (you can save them as a CSV or any other format)
df['eng_embedding'] = [emb.tolist() for emb in english_embeddings]
df.to_csv('english_embeddings.csv', index=False)
