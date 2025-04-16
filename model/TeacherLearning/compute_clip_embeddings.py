import torch
from tqdm import tqdm
import clip
from transformers import AutoTokenizer
from multilingual_clip import pt_multilingual_clip
import transformers
import torch.optim as optim
import os
import numpy as np

# ----------------------------------------------
# Load the pre-trained CLIP model (Teacher)
# ----------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using", device)
clip_model, preprocess = clip.load("ViT-B/32", device)

# ----------------------------------------------
# Load or compute English embeddings
# ----------------------------------------------
embeddings_file = "english_embeddings.npz"
df = pd.read_csv("dataset.csv")

if os.path.exists(embeddings_file):
    print(f"Embeddings file '{embeddings_file}' found. Loading embeddings...")
    english_embeddings = np.load(embeddings_file)["embeddings"]

    if np.isnan(english_embeddings).any():
        raise ValueError("❌ English embeddings contain NaN values!")
    if np.isinf(english_embeddings).any():
        raise ValueError("❌ English embeddings contain Inf values!")
    
    print("✅ LOADED SUCCESSFULLY – No NaN or Inf found in embeddings.")
else:
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
    print(f"Embeddings computed and saved to '{embeddings_file}'.")

# ----------------------------------------------
# Load the Multilingual CLIP model (Student)
# ----------------------------------------------
model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
multilingual_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
multilingual_model.to(device)

# ----------------------------------------------
# Prepare the Mongolian translations
# ----------------------------------------------
mongolian_sentences = df["mon"].tolist()

# ----------------------------------------------
# Optimizer and AMP setup
# ----------------------------------------------
optimizer = optim.Adam(multilingual_model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler()

def get_optimal_batch_size():
    batch_size = 256
    while batch_size > 1:
        try:
            torch.randn(batch_size, 512).cuda()
            return batch_size
        except RuntimeError:
            batch_size //= 2
            torch.cuda.empty_cache()
    return 1

BATCH_SIZE = 128
print(f"✅ Using Batch Size: {BATCH_SIZE}")

# ----------------------------------------------
# Training Loop
# ----------------------------------------------
multilingual_model.train()
for epoch in range(10):  # 10 epochs
    optimizer.zero_grad()
    
    with tqdm(total=len(mongolian_sentences), desc=f"Epoch {epoch + 1}", ncols=100) as pbar:
        for i in range(0, len(mongolian_sentences), BATCH_SIZE):
            batch_mongolian = mongolian_sentences[i: i + BATCH_SIZE]
            batch_english_np = english_embeddings[i: i + BATCH_SIZE]
            
            if np.isnan(batch_english_np).any() or np.isinf(batch_english_np).any():
                print(f"❌ Skipping batch {i} due to NaN/Inf in embeddings.")
                continue

            batch_english = torch.tensor(batch_english_np, dtype=torch.float32).squeeze(1).to(device)
            
            try:
                with torch.amp.autocast(device_type=device):
                    batch_student = multilingual_model(batch_mongolian, tokenizer).to(device)

                    # Cosine similarity loss
                    similarity = torch.nn.functional.cosine_similarity(batch_student, batch_english, dim=-1)
                    loss = 1 - similarity.mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ NaN or Inf loss detected at batch {i}, skipping...")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(multilingual_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                pbar.update(BATCH_SIZE)
                pbar.set_postfix(loss=loss.item())

            except Exception as e:
                print(f"⚠️ Exception at batch {i}: {e}")
                torch.cuda.empty_cache()
                continue

    print(f"✅ Epoch {epoch + 1} finished.")

# ----------------------------------------------
# Save the Fine-Tuned Model
# ----------------------------------------------
save_path = "fine_tuned_multilingual_clip"
multilingual_model.save_pretrained(save_path)
print(f"✅ Model saved to '{save_path}'")