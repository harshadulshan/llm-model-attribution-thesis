import pandas as pd
from sentence_transformers import SentenceTransformer, util

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', token=False)

print("Loading rewritten dataset...")
df = pd.read_csv("master_rewritten_samples_v2.csv")

cos_light_list = []
cos_heavy_list = []

print(f"Computing cosine similarity for {len(df)} rows...")
for idx, row in df.iterrows():
    orig = str(row['original_text'])
    light = str(row['light_rewrite'])
    heavy = str(row['heavy_rewrite'])
    embeddings = embedder.encode([orig, light, heavy])
    cos_light = float(util.cos_sim(embeddings[0], embeddings[1]))
    cos_heavy = float(util.cos_sim(embeddings[0], embeddings[2]))
    cos_light_list.append(cos_light)
    cos_heavy_list.append(cos_heavy)
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{len(df)} done...")

df['cos_light'] = cos_light_list
df['cos_heavy'] = cos_heavy_list
df['cos_pass_light'] = df['cos_light'] >= 0.80
df['cos_pass_heavy'] = df['cos_heavy'] >= 0.597
df['final_pass_light'] = df['pass_light'] & df['cos_pass_light']
df['final_pass_heavy'] = df['pass_heavy'] & df['cos_pass_heavy']
df.to_csv("master_rewritten_samples_FINAL.csv", index=False)

print("\n--- FINAL DUAL-METRIC GATE RESULTS ---")
print(f"Light: Lev-only = {df['pass_light'].mean()*100:.1f}% | Cos-only = {df['cos_pass_light'].mean()*100:.1f}% | BOTH = {df['final_pass_light'].mean()*100:.1f}%")
print(f"Heavy: Lev-only = {df['pass_heavy'].mean()*100:.1f}% | Cos-only = {df['cos_pass_heavy'].mean()*100:.1f}% | BOTH = {df['final_pass_heavy'].mean()*100:.1f}%")
print("\n--- Final pass rate by model ---")
print(df.groupby('model')[['final_pass_light','final_pass_heavy']].mean())
print("\n--- Final pass rate by domain ---")
print(df.groupby('domain')[['final_pass_light','final_pass_heavy']].mean())
print("\nSaved to master_rewritten_samples_FINAL.csv")
