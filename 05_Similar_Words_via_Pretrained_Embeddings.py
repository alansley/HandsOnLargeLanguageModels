import gc
import gensim.downloader as api

# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
# Other options include: "word2vec-google-news-300"
# Note: GLOVE stands for "Gloval Vectors for Word Representation", see: https://nlp.stanford.edu/projects/glove/
# Also: More options at https://github.com/RaRe-Technologies/gensim-data
model = api.load("glove-wiki-gigaword-50")

target_word: str = "king"
get_closest_count: int = 10

# Look up the 10 words most similar to our target word
most_similar_words = model.most_similar([model[target_word]], topn=get_closest_count)

# Print them in a clean, aligned table
print(f"{'Word':<15} | Similarity")
print("-" * 30)
for word, similarity_score in most_similar_words:
    print(f"{word:<15} | {similarity_score:.4f}")

# Clean up
gc.collect()
