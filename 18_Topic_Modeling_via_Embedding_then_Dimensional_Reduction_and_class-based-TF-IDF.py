# Note: This one starts of like `17_Text_Clustering_via_Embedding_then_Dimensional_Reduction` but rather than doing just
# the clustering we use BERTopic and class-based-TF-IDF (Term-Frequency / Inverse-Document-Frequency) analysis to
# generate the topics of the clusters. See p146-155.
#
# The full pipline goes:
# 1.) Clustering:
#     a.) SBERT
#     b.) SBERT
#     c.) SBERT
# 2.) Topic Representation:
#     a.) CountVectorizer (to generate the term frequency (TF))
#     b.) c-TF-IDF

from bertopic import BERTopic
from datasets import load_dataset, Dataset, load_from_disk
from hdbscan import HDBSCAN
import os
import plotly.io as pio
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
embeddings = embedding_model.encode(abstracts, show_progress_bar=True, device="cuda")

# Show that we got 44,949 abstracts each using an embedding of 384 values
print(f"\n--- Created embeddings for {embeddings.shape[0]} abstracts, where each embedding has {embeddings.shape[1]} values.")

# Now, before we cluster the abstracts, we'll reduce the dimensionality of the embeddings so we get more meaningful
# clusters. Note: In this code we'll use the UMAP (Uniform Manifold Approximation and Projection) technique, as it
# handles non-linear relationships and structures better than PCA (Principal Component Analysis):
# Further UMAP reading: https://arxiv.org/abs/1802.03426

# Reduce 384-dimensional embeddings to 2 dimensions for easier visualization
reduced_dimension_count = 2
print(f"\n--- Creating UMAP model to reduce dimensions from {embeddings.shape[1]} to {reduced_dimension_count}...")
umap_model = UMAP(
	n_components=reduced_dimension_count,
	min_dist=0.0,
	metric='cosine',
	random_state=42,
	verbose=True
)
reduced_embeddings = umap_model.fit_transform(embeddings)

# Confirm this shape
print("DEBUG!!!!!!")
print(reduced_embeddings.shape)  # Should be (n_docs, 2)
print(len(titles))               # Should match n_docs

# Now use HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to perform clustering.
# As a density-based method, HDBSCAN can also detect outliers in the data, which are data points that do not belong to
# any cluster. These outliers will not be assigned or forced to belong to any cluster. In other words, they are ignored.
# Since our ArXiv articles might contain some niche papers, using a model that detects outliers could be helpful.

# We fit the model and extract the clusters
print(f"\n--- Creating HDBSCAN model to map embeddings into clusters...")
hdbscan_model = HDBSCAN(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom").fit(reduced_embeddings)
clusters = hdbscan_model.labels_

# How many clusters did we generate? In this instance we're expecting 152 - which is quite a lot of clusters...
# Note: Because HDBSCAN uses outliers these will be topics 0..150 (e.g., 151 topics) and the special "-1" topic will be
# used for outliers that don't fit cleanly into any of our other topics.
num_clusters = len(set(clusters))
print(f"\n--- HDBSCAN generated {num_clusters} clusters.")

# ----- COMMENTED OUT
# # Print first three documents in cluster 0 - we expect them to be about sign-language in this instance!
# print("\n--- Our first few papers in the first cluster are all about sign-language!:")
# cluster = 0
# for index in np.where(clusters==cluster)[0][:3]:
# 	print(abstracts[index][:150] + "... \n")
#
# # Reduce 384-dimensional embeddings to two dimensions for easier visualization
# reduced_embeddings = UMAP(n_components=2, min_dist=0.0, metric="cosine", random_state=42).fit_transform(embeddings)
#
# # Create dataframe
# clusters_df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
# clusters_df["title"]   = titles
# clusters_df["cluster"] = [str(c) for c in clusters]
#
# # Select recognised and outlier clusters
# recognised_clusters_df = clusters_df.loc[clusters_df.cluster != "-1", :]
# outliers_df            = clusters_df.loc[clusters_df.cluster == "-1", :]
#
# # Plot recognised clusters (coloured) and outliers (grey) separately.
# # Note: Cluster colours get re-used because there's 161 of them but less than that in the colour-map - the take-away is
# # that there are lots of clusters and that they each have their own location and shape when crunched down to 2D space.
# plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
# plt.scatter(recognised_clusters_df.x, recognised_clusters_df.y, c=recognised_clusters_df.cluster.astype(int), alpha=0.6, s=2, cmap="tab20b")
# plt.axis("off")
#
# plt.show()
#
# # While this is visually interesting, it doesn't allow us to see what's happening inside the clusters. So rather than
# # stopping here, we can extend this visualization by going from TEXT clustering to TOPIC modeling.
# ----- END OF COMMENTED OUT


# Train our model with our previously defined models
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(abstracts, embeddings)

# info_df: DataFrame = topic_model.get_topic_info()
# print("ACL11111")
# print("Shape: " + info_df.shape)
# print(info_df)
# print("ACL22222")

#print(topic_model.get_topic_info())

print("\n--- The first topic just happens to be about speech (note: ASM is 'Automatic Speech Recognition')")
print(topic_model.get_topic(0))

# We can find topics by keyword - the result will look like:
# (
#  [7, 89, 96, 62, 44],
#  [0.84470475, 0.833554, 0.832743, 0.83094364, 0.8301422]
# )
#
# This tells us that topic 7 (first value in first array) has the most similarity to the word "graphics", at 0.84470475.
graphics_topic_query = topic_model.find_topics("graphics", top_n=5)
print(f"\n--- The most relevant topic to graphics is:\n{graphics_topic_query}")

# Then, if we grab the graphics topics we can see the top few words along with their c-TF-IDF relevance scores, ours
# will look like this:
# [
#   ('image', 0.02606524471683291),
#   ('visual', 0.02410945038705967),
#   ('vision', 0.014060394129393715),
#   ('images', 0.013634944817284174),
#   ('multimodal', 0.013583144955364362),
#   ('vqa', 0.012274565978217042),
#   ('modal', 0.011025970995319075),
#   ('captions', 0.011014083892998319),
#   ('captioning', 0.009682865901064713),
#   ('caption', 0.008365740881928098)
# ]
graphics_topic_number = graphics_topic_query[0][0]
graphics_topic_details = topic_model.get_topic(graphics_topic_number)
print(f"\n--- Details for topic {graphics_topic_number} are:\n{graphics_topic_details}")

# BERTopic uses Plotly for interactive plotting - so we'll set it to open each plot in our default browser.
# Note: Without this `show()`-ing a figure just prints the details to the console!
pio.renderers.default = "browser"

# Visualize topics and documents
documents_fig = topic_model.visualize_documents(
    docs=abstracts,
    reduced_embeddings=reduced_embeddings,
    hide_annotations=True
)

# Replace the hover text with the corresponding titles. Note: The scatter is in fig.data[0]
titles_as_hover = ["Title: " + title for title in titles]
documents_fig.data[0].hovertext = titles_as_hover
documents_fig.data[0].hovertemplate = "%{hovertext}<extra></extra>"

# Colour each title by topic in the legend
topics = topic_model.topics_
documents_fig.data[0].marker.color = topics

# Now show the plot!
documents_fig.show()

# Visualize barchart with ranked keywords
bar_fig = topic_model.visualize_barchart(top_n_topics=num_clusters)
bar_fig.show()

# Visualize relationships between topics. We have 161 topics (i.e., "clusters" - but we'll just show the first 30
heatmap_fig = topic_model.visualize_heatmap(width=1024,height=1024,n_clusters=30)
heatmap_fig.show()

# Visualize the potential hierarchical structure of topics. We'll only show the top 10 most frequent topics
hierarchy_fig = topic_model.visualize_hierarchy(top_n_topics=11)
hierarchy_fig.show()
