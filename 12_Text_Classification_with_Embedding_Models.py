import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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

# Convert our movie review text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings  = model.encode(data["test"]["text"], show_progress_bar=True)

# Our embeddings for the `train` split of our dataset will be of size (8530, 768) - this tells us that for each movie
# review in our dataset (there are 8,530) we have generated a 768 value tensor describing its embedding value / meaning.
print(f"\n--- Train embeddings shape is: {train_embeddings.shape}")  #

# Train a logistic regression on our train embeddings
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

# Next, let’s evaluate our model, i.e., predict previously unseen instances
y_predictions = clf.predict(test_embeddings)

def evaluate_performance(y_true, y_pred):
	"""Create and print the classification report"""
	performance = classification_report(y_true, y_pred, target_names=["Negative Review", "Positive Review"])
	print(f"\n--- Performance results:\n{performance}")

# Evaluate the model's ability to classify positive and negative movie reviews
evaluate_performance(data["test"]["label"], y_predictions)

# When evaluating our performance of classifying a movie review, there are 4 possible outcomes:
#   - TRUE POSITIVE  (TP) - The movie review is positive and we correctly classified it as positive,
#   - FALSE POSITIVE (FP) - The movie review is negative but we incorrectly classified it as positive,
#   - TRUE NEGATIVE  (TN) - The movie review is negative and we correctly classified it as negative, and
#   - FALSE NEGATIVE (FN) - The movie review is positive but we incorrectly classified it as negative.
#
# Running our `evaluate_performance` function will create a report like the following:
#
#                  precision    recall  f1-score   support
#
# Negative Review       0.85      0.86      0.85       533
# Positive Review       0.86      0.85      0.85       533
#
#        accuracy                           0.85      1066
#       macro avg       0.85      0.85      0.85      1066
#    weighted avg       0.85      0.85      0.85      1066
#
# So what does this all mean?:
#   - Precision measures how many of the items found are relevant, which indicates the accuracy of the relevant results.
#   - Recall refers to how many relevant classes were found, which indicates its ability to find all relevant results.
#   - Accuracy refers to how many correct predictions the model makes out of all predictions, which indicates the
#     overall correctness of the model, and the
#   - F1 Score balances both precision and recall to create a model's overall performance.
#
# Using our representation model we got a F1-Score of 0.81 (which is good), but using this embedding model we get an F1
# score of 0.85, which is even better!
