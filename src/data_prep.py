import pandas as pd
from datasets import Dataset
import os

def load_cola():
    # Download if not present
    if not os.path.exists('in_domain_train.tsv'):
        os.system('wget https://nyu-mll.github.io/CoLA/cola_public_1.1.zip')
        os.system('unzip -o cola_public_1.1.zip')
    
    # Logic from your notebook: CoLA has 4 columns, we need 'sentence' and 'label'
    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, 
                     names=['source', 'label', 'notes', 'sentence'])
    
    # Convert to Hugging Face Dataset format
    return Dataset.from_pandas(df[['sentence', 'label']])