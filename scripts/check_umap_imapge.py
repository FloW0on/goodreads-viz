import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("C:/book_atlas/dataset/processed/umap_n100000_seed42/umap2d_n100000_seed42.parquet")
plt.scatter(df['x'], df['y'], s=0.1, alpha=0.3)
plt.savefig("umap_check.png", dpi=150)