import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Optional

class EmbeddingPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the embedding pipeline.
        :param model_name: HuggingFace model name (e.g., 'all-MiniLM-L6-v2' or 'all-mpnet-base-v2')
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model '{model_name}' on device: {self.device.upper()}")
        
        self.model = SentenceTransformer(model_name, device=self.device)

    def compute_embeddings(
        self, 
        df: pd.DataFrame, 
        column: str, 
        batch_size: int = 64, 
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Computes embeddings for a specific column in the dataframe.
        
        :param df: The input pandas DataFrame.
        :param column: The column name to embed (e.g., 'title').
        :param batch_size: Number of sentences to process at once (adjust based on GPU VRAM).
        :param save_path: Optional path to save the .npy file.
        :return: A numpy array of shape (num_samples, embedding_dimension).
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        print(f"Preparing data for column: {column}")
        
        # 1. Handle Missing Values: Replace NaNs with empty strings or a placeholder
        # 2. Force conversion to string (to avoid errors with mixed types)
        sentences = df[column].fillna("").astype(str).tolist()

        print(f"Starting embedding computation for {len(sentences)} items...")
        
        # Compute embeddings
        embeddings = self.model.encode(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )

        print(f"Computation complete. Embedding shape: {embeddings.shape}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, embeddings)
            print(f"Embeddings saved to: {save_path}")

        return embeddings

if __name__ == "__main__":
    
    pass