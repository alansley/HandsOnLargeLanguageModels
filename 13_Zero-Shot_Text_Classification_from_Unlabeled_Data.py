import os
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Load our data via HuggingFace's `datasets` library. In this case it's an equal number of positive and negative movie
# reviews (I'm seeing 4,265 of each). Src: https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
dataset_name = "rotten_tomatoes"
cached_dataset_uri = "DataFiles/" + dataset_name
data = load_dataset(dataset_name)

# Save the dataset if we haven't already so we can automatically reload from our cached copy next time
if not os.path.exists(cached_dataset_uri):
	data.save_to_disk(cached_dataset_uri)

# The data is broken up into `train`, `validation` and `test` datasets as follows:
# DatasetDict({
#   train: Dataset({
#       features: ['text', 'label'],
#       num_rows: 8530
#   })
#   validation: Dataset({
#       features: ['text', 'label'],
#        num_rows: 1066
#   })
#   test: Dataset({
#     features: ['text', 'label'],
#       num_rows: 1066
#   })
# })
print(data)

reviews = data['train']
print(f"\n--- Total number of reviews in the 'train' set: {len(reviews)}")

# A label of 1 means it's a positive review of the movie (great!) while 0 means it's a negative review (sucked!)
positive_label = 1
negative_label = 0

# Count 'em up & show the stats
positive_count = 0
negative_count = 0
for review in reviews:
	if review['label'] == positive_label:
		positive_count += 1
	elif review['label'] == negative_label:
		negative_count += 1
	else:
		print(f"Got an unrecognised label value of {review['label']} - ignoring!")

print(f"\n--- To be specific, we have {positive_count} positive reviews and {negative_count} negative reviews.")

# Print the first review as a test
first_review = reviews[0]
print(f"\n--- First review: {first_review}")

# Load model.
# Note: We do NOT modify the embeddings in this embedding model - they are FROZEN at the values derived when this model was trained!
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Convert our test data into embeddings...
# Note: This is JUST the movie review text WITHOUT any classification of whether it's positive or negative!
test_embeddings  = model.encode(data["test"]["text"], show_progress_bar=True)

# ...then create embeddings for the labels that we're going to zero-shot classify against.
initial_labels = ["A negative review", "A positive review"]
initial_label_embeddings = model.encode(initial_labels)
print(f"\n--- Providing zero-shot classification labels of: {initial_labels}")

def test_label_embeddings(label_embeddings):
	# Now find the best matching label for each document via cosine similarity
	sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
	y_prediction = np.argmax(sim_matrix, axis=1)

	def evaluate_performance(y_true, y_pred):
		"""Create and print the classification report"""
		performance = classification_report(y_true, y_pred, target_names=["Negative Review", "Positive Review"])
		print(f"\n--- Performance results:\n\n{performance}")

	evaluate_performance(data["test"]["label"], y_prediction)

test_label_embeddings(initial_label_embeddings)

# When evaluating our performance of classifying a movie review, there are 4 possible outcomes:
#   - TRUE POSITIVE  (TP) - The movie review is positive - and we correctly classified it as positive,
#   - FALSE POSITIVE (FP) - The movie review is negative - but we incorrectly classified it as positive,
#   - TRUE NEGATIVE  (TN) - The movie review is negative - and we correctly classified it as negative, and
#   - FALSE NEGATIVE (FN) - The movie review is positive - but we incorrectly classified it as negative.
#
# Running our `evaluate_performance` function will create a report like the following:
#
#                  precision    recall  f1-score   support
#
# Negative Review       0.78      0.77      0.78       533
# Positive Review       0.77      0.79      0.78       533
#
#        accuracy                           0.78      1066
#       macro avg       0.78      0.78      0.78      1066
#    weighted avg       0.78      0.78      0.78      1066
#
# So what does this all mean?:
#   - Precision measures how many of the items found are relevant, which indicates the accuracy of the relevant results.
#   - Recall refers to how many relevant classes were found, which indicates its ability to find all relevant results.
#   - Accuracy refers to how many correct predictions the model makes out of all predictions, which indicates the
#     overall correctness of the model, and the
#   - F1 Score balances both precision and recall to create a model's overall performance.
#
# Using our zero-shot approach we got an F1 Score accuracy of 0.78 which ALMOST MATCHED the representation model's
# accuracy (0.80) - and we didn't even use any of the labelled data! We just created reasonable labels ("A positive
# review" / "A negative review") - then used the cosine similarity of the embedding values of those labels to the actual
# review text embeddings to classify the data!
#
# We didn't do QUITE as well as the 0.85 accuracy we got when USING the labeled training data via an embedding model -
# but we still did really well just by making two simple labels and comparing against their embeddings!
#
# However, we can go even further and try to make the labels more focussed by adding the word "movie", then we can push
# them out to their positive/negative extremes by adding the word "very" - so we get:
adjusted_labels = ["A very negative movie review", "A very positive movie review"]
adjusted_label_embeddings = model.encode(adjusted_labels)
print(f"\n--- Now adjusting those labels to: {adjusted_labels}")

test_label_embeddings(adjusted_label_embeddings)

# With that done, we get an average F1 Score accuracy of 0.80 which MATCHES the representation model result (which was
# USING the labeled training data - and actually BEATS IT by 0.01 on positive review accuracy) - and all this while we
# were just zero-shot-ing it from the above two labels that we came up with ourselves! Nice! =D
#
# --- Performance results:
#
#                  precision    recall  f1-score   support
#
# Negative Review       0.86      0.73      0.79       533
# Positive Review       0.76      0.88      0.82       533
#
#        accuracy                           0.80      1066
#       macro avg       0.81      0.80      0.80      1066
#    weighted avg       0.81      0.80      0.80      1066
