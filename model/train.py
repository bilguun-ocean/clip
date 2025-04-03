import pandas as pd
import torch
import clip
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for the progress bar
from transformers import AutoTokenizer
from multilingual_clip import pt_multilingual_clip

# Step 1: Load your CSV file
df = pd.read_csv('dataset.csv')

# Step 2: Prepare CLIP model to get English text embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Get English sentences and compute CLIP embeddings
english_sentences = df['eng'].tolist()  # English sentences from the 'eng' column
text_inputs = clip.tokenize(english_sentences, truncate=True).to(device)

with torch.no_grad():
    english_embeddings = clip_model.encode_text(text_inputs)

print(f"English Text Embeddings shape: {english_embeddings.shape}")

# Step 3: Load the Multilingual CLIP model (XLM-Roberta-Large-Vit-B-32)
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
multilingual_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Tokenize Mongolian translations (from the 'mon' column)
mongolian_sentences = df['mon'].tolist()  # Mongolian translations from the 'mon' column
mongolian_inputs = tokenizer(mongolian_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Move to the correct device (GPU/CPU)
mongolian_inputs = {key: value.to(device) for key, value in mongolian_inputs.items()}

# Step 5: Define optimizer
optimizer = optim.Adam(multilingual_model.parameters(), lr=1e-5)

# Step 6: Training loop (with Teacher Learning)
multilingual_model.train()

# Use tqdm for the progress bar in the loop
for epoch in range(3):  # Example: 3 epochs for training
    optimizer.zero_grad()
    
    # Add tqdm here to show progress in each epoch
    with tqdm(total=len(mongolian_sentences), desc=f'Epoch {epoch + 1}', ncols=100) as pbar:
        for i in range(0, len(mongolian_sentences), 32):  # Mini-batch processing (adjust batch size as needed)
            batch_inputs = {key: value[i:i+32] for key, value in mongolian_inputs.items()}  # Slice the batch
            batch_student_embeddings = multilingual_model(**batch_inputs)  # Forward pass through the student model
            
            # Calculate contrastive loss (using cosine similarity between student and teacher embeddings)
            loss = torch.nn.functional.cosine_similarity(batch_student_embeddings, english_embeddings[i:i+32], dim=-1).mean()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.update(32)
            pbar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Step 7: Save the fine-tuned model
multilingual_model.save_pretrained("fine_tuned_multilingual_clip")
