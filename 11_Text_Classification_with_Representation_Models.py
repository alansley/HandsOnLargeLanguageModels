import os
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

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

# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model into pipeline
pipe = pipeline(
	model=model_path,
	tokenizer=model_path,
	return_all_scores=True,
	device="cuda:0"
)

# Run inference on the "test" split of our movie review dataset
y_prediction = []
test_split_of_dataset = KeyDataset(data["test"], key="text")
for output in tqdm(pipe(test_split_of_dataset), total=len(data["test"])):
	negative_score = output[0]["score"]
	positive_score = output[2]["score"]
	assignment = np.argmax([negative_score, positive_score])
	y_prediction.append(assignment)

# Now that we have generated our predictions, all that is left is evaluation. We create a small function that we can
# easily use throughout this chapter:

def evaluate_performance(y_true, y_pred):
	"""Create and print the classification report"""
	performance = classification_report(y_true, y_pred, target_names=["Negative Review", "Positive Review"])
	print(f"\n--- Performance results:\n\n{performance}")

evaluate_performance(data["test"]["label"], y_prediction)

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
# Negative Review       0.76      0.88      0.81       533
# Positive Review       0.86      0.72      0.78       533
#
#        accuracy                           0.80      1066
#       macro avg       0.81      0.80      0.80      1066
#    weighted avg       0.81      0.80      0.80      1066
#
# So what does this all mean?:
#   - Precision measures how many of the items found are relevant, which indicates the accuracy of the relevant results.
#   - Recall refers to how many relevant classes were found, which indicates its ability to find all relevant results.
#   - Accuracy refers to how many correct predictions the model makes out of all predictions, which indicates the
#     overall correctness of the model, and the
#   - F1 Score balances both precision and recall to create a model's overall performance.
#
# Apparently, an F1 Score 0.80 is a pretty good classification result for a model not specifically trained on the domain
# data.
#
# See p119-p121 for more details - honestly, I don't feel this is particularly well explained - but with any luck it'll
# make more sense later.
