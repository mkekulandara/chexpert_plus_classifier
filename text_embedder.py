import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read the CSV file
df = pd.read_csv('/Users/madhukarakekulandara/Downloads/gen ai project/chexpert_plus_classifier/data/df_chexpert_plus_240401_small.csv')

# Function to extract the impression from the report
def extract_impression(report):
    # Find the impression section
    impression_match = re.search(r'IMPRESSION:\s*(.*?)(?:\n\n|\Z)', report, re.DOTALL | re.IGNORECASE)
    if impression_match:
        return impression_match.group(1).strip()
    return ''

# Apply the extraction function to the 'report' column
df['impression'] = df['report'].apply(extract_impression)

# Create a new DataFrame with only 'path_to_image' and 'impression'
result_df = df[['path_to_image', 'impression']]

# Define the models to use
models = {
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biobert": "dmis-lab/biobert-v1.1",
    "biogpt": "microsoft/biogpt",
    "bluebert": "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
}

# Load models and tokenizers
tokenizers = {}
model_instances = {}
for model_name, model_path in models.items():
    try:
        tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
        model_instances[model_name] = AutoModel.from_pretrained(model_path)
        logger.info(f"Successfully loaded {model_name}")
    except ImportError as e:
        logger.warning(f"Failed to load {model_name}: {str(e)}")
        logger.warning(f"Skipping {model_name}. To use this model, please install the required dependencies.")
    except Exception as e:
        logger.error(f"An error occurred while loading {model_name}: {str(e)}")
        logger.warning(f"Skipping {model_name}")

# Function to create embeddings
def create_embedding(text, model_name):
    tokenizer = tokenizers[model_name]
    model = model_instances[model_name]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply the embedding function to the 'impression' column for each successfully loaded model
for model_name in tokenizers.keys():
    logger.info(f"Creating embeddings for {model_name}")
    result_df[f'{model_name}_embedding'] = result_df['impression'].apply(lambda x: create_embedding(x, model_name))

# Free up memory
del tokenizers, model_instances
torch.cuda.empty_cache()

print(result_df.head())
print(result_df.columns)

# Get the directory of the input CSV file
input_dir = os.path.dirname('/Users/madhukarakekulandara/Downloads/gen ai project/chexpert_plus_classifier/data/')

# Construct the path for the output CSV file
output_path = os.path.join(input_dir, 'text_embeddings.csv')

# Save the result_df to the new CSV file
result_df.to_csv(output_path, index=False)

print(f"DataFrame saved to: {output_path}")
