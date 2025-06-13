from datasets import load_dataset

# Load our data via HuggingFace's `datasets` library. In this case it's 5,331 positive movie reviews & 5,331 negative
# reviews - see: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
# Note: The datasets lib will cache this file on first download so we don't have to manually cache it to prevent
# re-download per run.
data = load_dataset("rotten_tomatoes")

