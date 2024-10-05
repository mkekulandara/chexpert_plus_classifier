import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# Load the BioBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
biobert_model = TFBertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1", from_pt=True)

def generate_embedding(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding='max_length')
    
    # Generate embeddings using BioBERT
    outputs = biobert_model(**inputs)
    
    # Get the embeddings for the [CLS] token (this can be used as a summary of the text)
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding

# Example usage with a radiology report
radiology_report = """
Patient presents with mild chest pain. The CT scan shows no significant anomalies in the lungs, but there is a small nodule in the left lower lobe, likely benign. 
Further examination recommended to rule out malignancy.
"""

# Generate the embedding
embedding = generate_embedding(radiology_report)

# Convert the embedding to a NumPy array for further use
embedding_np = embedding.numpy()

# Print the shape of the embedding (should be [1, 768])
print("Embedding shape:", embedding_np.shape)
print("Embedding vector:", embedding_np)
