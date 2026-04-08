import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Optional

class NomicEmbeddingPipeline:
    def __init__(self, model_name: str = 'nomic-ai/nomic-embed-text-v1.5'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Nomic model on {self.device.upper()}...")
        
        # 1. CRITICAL: Nomic requires trust_remote_code=True
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)

    def compute_embeddings(
        self, 
        df: pd.DataFrame, 
        column: str, 
        batch_size: int = 64, 
        save_path: Optional[str] = None
    ) -> np.ndarray:
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")

        # 2. TASK PREFIX: Nomic v1.5 requires a prefix for best performance.
        # For node features in a graph, 'search_document: ' is the standard.
        print(f"Preparing data with 'search_document:' prefix...")
        sentences = [f"search_document: {str(x)}" for x in df[column].fillna("missing_data")]

        print(f"Starting computation for {len(sentences)} items...")
        
        # 3. COMPUTE: Same as your original pipeline
        embeddings = self.model.encode(
            sentences, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, embeddings)
            print(f"Embeddings saved to: {save_path}")

        return embeddings