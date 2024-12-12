import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the DataFrame
df = pd.read_csv("partition_4.csv")

# Base path for the images
img_base_path = '/scratch/workspace/mkekulandara_uri_edu-imgspace/chexpert/PNG/'

# Step 1 & 2: Modify file extension and create a new DataFrame
new_df = pd.DataFrame()
new_df['path_to_image'] = df['path_to_image'].copy()

# Load the pretrained ViT model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to get image embedding
def get_image_embedding(image_path):
    try:
        # Open and transform the image
        img = Image.open(img_base_path + image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Get features
        with torch.no_grad():
            features = feature_extractor(images=img_tensor, return_tensors="pt")
            outputs = model(**features)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        # Return the embedding as a list
        return embedding.tolist()
    except Exception as e:
        print(f"Error processing {img_base_path + image_path}: {str(e)}")
        return None

# Apply the function to each image path with a progress bar
tqdm.pandas()  # Enable tqdm for pandas
new_df['image_embedding'] = new_df['path_to_image'].progress_apply(get_image_embedding)

# Save the updated DataFrame
new_df.to_csv('partition_4_img_emd.csv', index=False)
