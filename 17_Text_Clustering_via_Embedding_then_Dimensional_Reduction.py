import os
from datasets import load_dataset, Dataset, load_from_disk
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP  # Note: We install the "umap-learn" package rather than just "umap"

# Load data from Hugging Face
dataset_name = "maartengr/arxiv_nlp"
slice_name = "train"
cached_dataset_uri = "DataFiles/" + dataset_name

def load_data(slice_name_arg: str = None):
	# Get the data from hugging face if we don't already have it then save it to our cache
	# Note: If a specific slice was specified then we only grab that slice.
	if not os.path.exists(cached_dataset_uri):
		if slice_name is not None:
			data: Dataset = load_dataset(dataset_name)[slice_name_arg]
		else:
			data: Dataset = load_dataset(dataset_name)
		data.save_to_disk(cached_dataset_uri)
		return data
	else:
		# Otherwise if we already have a cached copy then load that rather than re-downloading.
		# Note: If we previously only downloaded a specific slice we don't have to specify it as that's all we have!
		data: Dataset = load_from_disk(cached_dataset_uri)
		return data

# Download the dataset on first run, then save it & use the cached version in future runs
dataset: Dataset = load_data(slice_name)

# Extract metadata
abstracts = dataset["Abstracts"]
titles    = dataset["Titles"]

print("--- First record:")
print(f"Title: {titles[0]}")
print(f"Abstract: {abstracts[0]}")

# Create an embedding for each abstract.
# Note: Generating these embeddings takes ~1 min on a RTX 4090 so will likely take several minutes if running CPU-only!
embedding_model_name = "thenlper/gte-small"
embedding_model = SentenceTransformer(embedding_model_name)
print("\n--- Encoding abstracts as embeddings...")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

# Show that we got 44,949 abstracts each using an embedding of 384 values
print(f"\n--- Created embeddings for {embeddings.shape[0]} abstracts, where each embedding has {embeddings.shape[1]} values.")

# Now, before we cluster the abstracts, we'll reduce the dimensionality of the embeddings so we get more meaningful
# clusters. Note: In this code we'll use the UMAP (Uniform Manifold Approximation and Projection) technique, as it
# handles non-linear relationships and structures better than PCA (Principal Component Analysis):
# Further UMAP reading: https://arxiv.org/abs/1802.03426

# Reduce the input embeddings from 384 to 5 dimensions
umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

# Now use HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to perform clustering.
# As a density-based method, HDBSCAN can also detect outliers in the data, which are data points that do not belong to
# any cluster. These outliers will not be assigned or forced to belong to any cluster. In other words, they are ignored.
# Since our ArXiv articles might contain some niche papers, using a model that detects outliers could be helpful.

# We fit the model and extract the clusters
hdbscan_model = HDBSCAN(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom").fit(reduced_embeddings)
clusters = hdbscan_model.labels_

# How many clusters did we generate? In this instance we're expecting 161 - which is quite a lot of clusters..
print(f"\n--- HDBSCAN generated {len(set(clusters))} clusters.")

# Print first three documents in cluster 0 - we expect them to be about sign-language in this instance!
print("\n--- Our first few papers in the first cluster are all about sign-language!:")
cluster = 0
for index in np.where(clusters==cluster)[0][:3]:
	print(abstracts[index][:150] + "... \n")

# Reduce 384-dimensional embeddings to 2 dimensions for easier visualization
reduced_embeddings = UMAP(n_components=2, min_dist=0.0, metric="cosine", random_state=42).fit_transform(embeddings)

# Create dataframe
clusters_df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
clusters_df["title"]   = titles
clusters_df["cluster"] = [str(c) for c in clusters]

# Select recognised and outlier clusters
recognised_clusters_df = clusters_df.loc[clusters_df.cluster != "-1", :]
outliers_df            = clusters_df.loc[clusters_df.cluster == "-1", :]

# Plot recognised clusters (coloured) and outliers (grey) separately.
# Note: Cluster colours get re-used because there's 161 of them but less than that in the colour-map - the take-away is
# that there are lots of clusters and that they each have their own location and shape when crunched down to 2D space.
plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
plt.scatter(recognised_clusters_df.x, recognised_clusters_df.y, c=recognised_clusters_df.cluster.astype(int), alpha=0.6, s=2, cmap="tab20b")
plt.axis("off")

plt.show()

# While this is visually interesting, it doesn't allow us to see what's happening inside the clusters. So rather than
# stopping here, we can extend this visualization by going from TEXT clustering to TOPIC modeling.
