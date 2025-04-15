import pandas as pd
import torch
from tqdm import tqdm
import clip
from transformers import AutoTokenizer
import transformers
from multilingual_clip import pt_multilingual_clip
import torch.optim as optim
import os
import numpy as np

# Load the pre-trained CLIP model (Teacher)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using", device)
clip_model, preprocess = clip.load("ViT-B/32", device)

# Check if the embeddings file already exists
# embeddings_file = "english_embeddings.csv"
embeddings_file = "english_embeddings.npz"
df = pd.read_csv("dataset.csv")

if os.path.exists(embeddings_file):
    print(f"Embeddings file '{embeddings_file}' found. Loading embeddings...")
    # df = pd.read_csv(embeddings_file)
    # english_embeddings = [
    #     eval(emb) for emb in df["eng_embedding"]
    # ]  # Convert string back to list
    english_embeddings = np.load(embeddings_file)["embeddings"]
    print("LOADED SUCCESSFULLY")
else:
    # df = pd.read_csv("dataset.csv")
    # Function to compute text embeddings
    def compute_clip_embeddings(texts):
        embeddings = []
        for text in tqdm(texts, desc="Processing captions", unit="caption"):
            text_input = clip.tokenize([text], truncate=True).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_input)
            embeddings.append(text_features.cpu().numpy())
        return embeddings

    english_embeddings = compute_clip_embeddings(df["eng"].tolist())
    np.savez(embeddings_file, embeddings=np.array(english_embeddings))
    # df["eng_embedding"] = [emb.tolist() for emb in english_embeddings]
    # df.to_csv(embeddings_file, index=False)
    print(f"Embeddings computed and saved to '{embeddings_file}'.")

# Load the Multilingual CLIP model (Student)
model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
multilingual_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
multilingual_model.to(device)

# Tokenize Mongolian translations (from the 'mon' column)
mongolian_sentences = df["mon"].tolist()

# Define optimizer
optimizer = optim.Adam(multilingual_model.parameters(), lr=1e-5)


# 🔹 **Find the Optimal Batch Size** (Auto)
def get_optimal_batch_size():
    """Dynamically finds the largest batch size that fits in GPU memory."""
    batch_size = 16  # Start high for L100
    while batch_size > 1:
        try:
            test_tensor = torch.randn(batch_size, 512).cuda()
            del test_tensor
            return batch_size
        except RuntimeError:
            batch_size //= 2
            torch.cuda.empty_cache()
    return 1  # Minimum batch size


BATCH_SIZE = get_optimal_batch_size()
print(f"✅ Optimal Batch Size: {BATCH_SIZE}")

# Enable mixed precision (AMP) for speed & memory efficiency
scaler = torch.amp.GradScaler()

# Training loop
multilingual_model.train()
for epoch in range(3):  # Example: 3 epochs
    optimizer.zero_grad()

    with tqdm(
        total=len(mongolian_sentences), desc=f"Epoch {epoch + 1}", ncols=100
    ) as pbar:
        for i in range(0, len(mongolian_sentences), BATCH_SIZE):
            batch_mongolian_sentences = mongolian_sentences[i : i + BATCH_SIZE]
            batch_english_embeddings = torch.tensor(
                english_embeddings[i : i + BATCH_SIZE]
            ).to(device)

            with torch.amp.autocast(device_type=device):  # Mixed precision training
                batch_student_embeddings = multilingual_model.forward(
                    batch_mongolian_sentences, tokenizer
                ).to(device)
                loss = torch.nn.functional.cosine_similarity(
                    batch_student_embeddings, batch_english_embeddings, dim=-1
                ).mean()

            # Backpropagation with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update progress bar
            pbar.update(BATCH_SIZE)
            pbar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the fine-tuned model
multilingual_model.save_pretrained("fine_tuned_multilingual_clip")
print("✅ Model saved to 'fine_tuned_multilingual_clip'.")
