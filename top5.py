import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load CLIP caption embeddings
clip_data = np.load("caption_embeddings_clip.npz", allow_pickle=True)
clip_image_ids = clip_data["image_ids"]
clip_text_embeddings = clip_data["embeddings"]

# Load M-CLIP caption embeddings
mclip_data = np.load("caption_embeddings_mclip.npz", allow_pickle=True)
mclip_image_ids = mclip_data["image_ids"]
mclip_text_embeddings = mclip_data["embeddings"]

# Load image embeddings
image_data = np.load("image_embeddings.npz", allow_pickle=True)
image_filenames = image_data["file_names"]
image_embeddings = image_data["embeddings"]

# Match image ids between datasets: keep only those present in all three sets
matched_image_ids = [image_id for image_id in image_filenames if image_id in clip_image_ids and image_id in mclip_image_ids]

# Initialize lists for storing matched embeddings
matched_image_embeddings = []
matched_clip_text_embeddings = []
matched_mclip_text_embeddings = []

# Add a progress bar while processing matched image IDs
for image_id in tqdm(matched_image_ids, desc="Processing Image Embeddings"):
    image_idx = image_filenames.tolist().index(image_id)
    matched_image_embeddings.append(image_embeddings[image_idx])
    
    clip_idx = clip_image_ids.tolist().index(image_id)
    matched_clip_text_embeddings.append(clip_text_embeddings[clip_idx])
    
    mclip_idx = mclip_image_ids.tolist().index(image_id)
    matched_mclip_text_embeddings.append(mclip_text_embeddings[mclip_idx])

# Convert to numpy arrays for computation
matched_image_embeddings = np.array(matched_image_embeddings)
matched_clip_text_embeddings = np.array(matched_clip_text_embeddings)  # Shape: (N, 5, 512)
matched_mclip_text_embeddings = np.array(matched_mclip_text_embeddings)

# Accuracy calculation function
def compute_top5_accuracy(image_ids, text_embeddings):
    """
    image_ids: list or array of image ids (length N)
    text_embeddings: array of shape (N, num_caps, emb_dim), e.g. (31014, 5, 512)
    """
    correct = 0
    total = len(image_ids)
    print(f"Image embeddings shape: {matched_image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    print(f"Image IDs shape: {len(image_ids)}")

    # Flatten text embeddings to shape (N * num_caps, emb_dim)
    flat_text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-1])
    
    # Create a flattened list of image ids corresponding to each caption embedding.
    flat_text_image_ids = []
    for image_id, caps in zip(image_ids, text_embeddings):
        n_caps = caps.shape[0]  # number of captions per image
        flat_text_image_ids.extend([image_id] * n_caps)
    
    # For each image, compute cosine similarity against all caption embeddings
    for idx, image_name in tqdm(enumerate(image_ids), desc="Calculating Top-5 Accuracy", total=total):
        image_vector = matched_image_embeddings[idx].reshape(1, -1)  # shape (1, emb_dim)
        similarities = cosine_similarity(image_vector, flat_text_embeddings)[0]
        
        # Get indices for the top-5 highest similarities
        top_5_indices = similarities.argsort()[-5:][::-1]
        
        # Map these indices to image ids using the flattened list
        top_5_image_ids = [flat_text_image_ids[i] for i in top_5_indices]
        
        # Check if the correct image id is among the top-5
        if image_name in top_5_image_ids:
            correct += 1

    return correct / total

# Compute top-5 accuracy for CLIP and M-CLIP caption embeddings
clip_accuracy = compute_top5_accuracy(matched_image_ids, matched_clip_text_embeddings)
mclip_accuracy = compute_top5_accuracy(matched_image_ids, matched_mclip_text_embeddings)

print(f"CLIP Accuracy: {clip_accuracy:.4f}")
print(f"M-CLIP Accuracy: {mclip_accuracy:.4f}")
