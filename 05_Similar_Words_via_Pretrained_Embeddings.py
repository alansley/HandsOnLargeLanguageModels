import gensim.downloader as api

# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)
# Other options include "word2vec-google-news-300"
# Note: GLOVE stands for "Gloval Vectors for Word Representation", see: https://nlp.stanford.edu/projects/glove/
# Also: More options at https://github.com/RaRe-Technologies/gensim-data
model = api.load("glove-wiki-gigaword-50")

target_word: str = "king"
get_closest_count: int = 10

# Look up the 10 words most similar to our target word
most_similar_words = model.most_similar([model[target_word]], topn=get_closest_count)

# Print 'em
for word, simularity_score in most_similar_words:
	# Choose to use 1 or 2 tabs depending on the length of the simular word so that everything lines up nicely
	tab_count = 1
	if len(word) < 4:
		tab_count = 2
	tab_string: str = "\t" * tab_count

	print(word + tab_string + " - " + str(simularity_score))
