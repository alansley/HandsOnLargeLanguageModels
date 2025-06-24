# Example source:
# https://maartengr.github.io/BERTopic/getting_started/visualization/visualize_documents.html

from bertopic import BERTopic
import plotly.io as pio
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from umap import UMAP

# Prepare embeddings
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

# Train BERTopic
topic_model = BERTopic().fit(docs, embeddings)

# Run the visualization with the original embeddings
topic_model.visualize_documents(docs, embeddings=embeddings)

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

# BERTopic uses Plotly for interactive plotting - so we'll set it to open each plot in our default browser.
# Note: Without this `show()`-ing a figure just prints the details to the console!
pio.renderers.default = "browser"

# *** I HAVE NO IDEA WHY THIS DOESN'T WORK - SAME ISSUE AS 18a - WE CAN SEE THE AXIS LABELS BUT NOT THE PLOTTED DATA ***
topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings).show()