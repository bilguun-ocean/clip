import torch
import numpy as np
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# CLIP загвар болон процессорыг ачаалах
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Flickr30k датасетийг ачаалах
dataset = load_dataset("nlphuji/flickr30k", split="test")  # 'train', 'test', 'validation' байгаа

image_embeddings = []
file_names = []

# Бүх зураг дээр давтах
for item in tqdm(dataset):
    image = item["image"]
    file_name = item["filename"]

    # Зураг CLIP-д тохируулах
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Embedding авах
    with torch.no_grad():
        image_features = model.get_image_features(inputs["pixel_values"])

    # Embedding-ийг нормализаци хийх
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Массивт хадгалах
    image_embeddings.append(image_features.cpu().numpy())
    file_names.append(file_name)

# NPZ файлд хадгалах
np.savez("image_embeddings.npz", file_names=np.array(file_names), embeddings=np.vstack(image_embeddings))

print("NPZ файл хадгалагдлаа: image_embeddings.npz")
