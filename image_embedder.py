import tensorflow as tf
from PIL import Image
from transformers import ViTFeatureExtractor, TFAutoModel
import numpy as np

# Load the image
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

# Preprocess the image
def preprocess_image(image, feature_extractor):
    inputs = feature_extractor(images=image, return_tensors="tf")
    return inputs['pixel_values']

# Convert image to embedding using Hugging Face pretrained model
def image_to_embedding(image_path, model_name="google/vit-base-patch16-224"):
    # Load the feature extractor and the model
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    
    # Load and preprocess the image
    img = load_image(image_path)
    pixel_values = preprocess_image(img, feature_extractor)
    
    # Generate embeddings
    embeddings = model(pixel_values).last_hidden_state
    # Optionally, you can take the mean of the embeddings along the sequence length axis
    embeddings = tf.reduce_mean(embeddings, axis=1)
    
    return embeddings

# Example usage
image_path = '/content/drive/MyDrive/Gen AI class/data/000001-1.png'  # Replace with the path to your image
embedding = image_to_embedding(image_path)

# Convert embedding to numpy array and print its shape
embedding_np = embedding.numpy()
print("Embedding shape:", embedding_np.shape)
