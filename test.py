import torch
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained("fine_tuned_multilingual_clip")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")  # Load tokenizer
model.to(device)

# Test sentence in Mongolian
sentence = "Сайн уу, энэ загвар ажиллаж байна уу?"  # Mongolian test sentence

# Inference: passing raw text sentence to the model
with torch.no_grad():
    embedding = model.forward([sentence], tokenizer)  # Pass the raw text list directly

print("✅ Embedding shape:", embedding.shape)
print("✅ Sample embedding:", embedding[0][:5])  # Show first 5 dimensions of the embedding
