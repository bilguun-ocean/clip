from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel
import transformers
from multilingual_clip import pt_multilingual_clip

app = Flask(__name__)

# Load OpenAI CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load mCLIP model
mclip_model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"  # Adjust model name for mCLIP
mclip_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(mclip_model_name).to(device)
mclip_tokenizer = transformers.AutoTokenizer.from_pretrained(mclip_model_name)

# Load embeddings
clip_data = np.load("caption_embeddings_clip.npz", allow_pickle=True)
mclip_data = np.load("caption_embeddings_mclip.npz", allow_pickle=True)
image_data = np.load("image_embeddings.npz", allow_pickle=True)

clip_image_ids = clip_data["image_ids"]  # (M,) image ids for OpenAI CLIP
clip_embeddings = clip_data["embeddings"]  # (M, 512) text embeddings for OpenAI CLIP

mclip_image_ids = mclip_data["image_ids"]  # (M,) image ids for mCLIP
mclip_embeddings = mclip_data["embeddings"]  # (M, 512) text embeddings for mCLIP

image_embeddings = image_data["embeddings"]  # (N, 512) image embeddings
image_file_names = image_data["file_names"]  # (N,) image file names

def get_clip_embedding(text):
    """Generates a CLIP embedding for the given text input."""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs).cpu().numpy()
    return embedding

def get_mclip_embedding(text):
    """Generates an mCLIP embedding for the given text input."""
    inputs = mclip_tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=77,  # Use max length of 77 tokens
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        text_features = mclip_model.forward(text, mclip_tokenizer)
    
    # Normalize embeddings
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    user_caption = request.form['caption']
    model_choice = request.form.get('model_choice', 'clip')  # Get the model choice (clip or mclip)

    if model_choice == 'clip':
        caption_embedding = get_clip_embedding(user_caption)
        similarities = cosine_similarity(caption_embedding, image_embeddings)[0]
    elif model_choice == 'mclip':
        caption_embedding = get_mclip_embedding(user_caption)
        similarities = cosine_similarity(caption_embedding, image_embeddings)[0]

    # Get top-5 matches
    top5_indices = np.argsort(similarities)[-5:][::-1]
    top5_image_ids = [image_file_names[idx] for idx in top5_indices]

    return render_template('results.html', caption=user_caption, image_ids=top5_image_ids, model_choice=model_choice)

if __name__ == '__main__':
    app.run(debug=True)
