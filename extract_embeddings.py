import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', token=False)

print("Loading feature dataset...")
df = pd.read_csv("features_extracted.csv")

print(f"Encoding {len(df)} texts (384-dim embeddings)...")
texts = df['text'].astype(str).tolist()
embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)

# Save as a separate .npy file (matches row order of features_extracted.csv exactly)
np.save("semantic_embeddings.npy", embeddings)

print(f"\nDone! Shape: {embeddings.shape}")
print("Saved to semantic_embeddings.npy — row i corresponds to row i in features_extracted.csv")