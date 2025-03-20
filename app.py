from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

# Load OpenAI CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load embeddings
text_data = np.load("text_embeddings.npz", allow_pickle=True)
image_data = np.load("image_embeddings.npz", allow_pickle=True)

text_embeddings = text_data["caption"]  # (M, 512) text embeddings
text_file_names = text_data["file_name"]  # (M,) file names

image_embeddings = image_data["embeddings"]  # (N, 512) image embeddings
image_file_names = image_data["file_names"]  # (N,) image file names

def get_text_embedding(text):
    """Generates a CLIP embedding for the given text input."""
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs).cpu().numpy()
    return embedding

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    user_caption = request.form['caption']

    # Generate embedding using OpenAI CLIP (to match stored embeddings)
    caption_embedding = get_text_embedding(user_caption)

    # Compute cosine similarity
    similarities = cosine_similarity(caption_embedding, image_embeddings)[0]

    # Get top-5 matches
    top5_indices = np.argsort(similarities)[-5:][::-1]
    top5_image_ids = [image_file_names[idx] for idx in top5_indices]  

    return render_template('results.html', caption=user_caption, image_ids=top5_image_ids)

if __name__ == '__main__':
    app.run(debug=True)
