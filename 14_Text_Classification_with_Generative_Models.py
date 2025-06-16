import os
from datasets import load_dataset
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

# Now we'll ADJUST our data to create a version of each review which is prefixed with our prompt!
prompt = "Is the following sentence positive or negative? "  # Note the space on the end so it doesn't run into the review!
data = data.map(lambda og_review_txt: { "t5": prompt + og_review_txt['text']})

# The data is still broken up into `train`, `validation` and `test` datasets - but has this additional 't5' field for
# each record:
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label', 't5'],
#         num_rows: 8530
#     })
#     validation: Dataset({
#         features: ['text', 'label', 't5'],
#         num_rows: 1066
#     })
#     test: Dataset({
#         features: ['text', 'label', 't5'],
#         num_rows: 1066
#     })
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
print(f"\n--- First prompt prefixed review: {first_review['t5']}")

# Path to our HF model.
# Note: The "T5" comes from "Text-to-Text Transfer Transformer" - so it takes text in, and then spits text out.
# Also: The T5 architecture is a decoder-encoder architecture like the original Transformer model. It was trained using
# token masking (see p129 - figures 4.19 & 4.20).
# Further Reading: "Scaling Instruction-Finetuned Language Models" - https://arxiv.org/abs/2210.11416
model_path = "google/flan-t5-small"  # Other sizes available: "-base" / "-large" / "-xl" / "-xxl"

# Load model into pipeline - this time SPECIFICALLY to perform text generation
model_pipeline = pipeline(
	task="text2text-generation",
	model=model_path,
	device="cuda:0"
)

# Run inference on the "test" slice using our T5 key (prompt + review)
y_prediction = []
for output in tqdm(model_pipeline(KeyDataset(data["test"], "t5")), total=len(data["test"])):
	text = output[0]["generated_text"]
	y_prediction.append(0 if text == "negative" else 1)

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
# Running our `evaluate_performance` function creates the following report::
#
#                  precision    recall  f1-score   support
#
# Negative Review       0.83      0.85      0.84       533
# Positive Review       0.85      0.83      0.84       533
#
#        accuracy                           0.84      1066
#       macro avg       0.84      0.84      0.84      1066
#    weighted avg       0.84      0.84      0.84      1066
#
# So what does this all mean?:
#   - Precision measures how many of the items found are relevant, which indicates the accuracy of the relevant results.
#   - Recall refers to how many relevant classes were found, which indicates its ability to find all relevant results.
#   - Accuracy refers to how many correct predictions the model makes out of all predictions, which indicates the
#     overall correctness of the model, and the
#   - F1 Score balances both precision and recall to create a model's overall performance.
#
# At 0.84 we are just 0.01 below the F1 score result of our Text Classification with Embedding Models (which scored
# 0.85) - and that was using LABELED DATA!
#
# We also beat the F1 score for labeled data using representation models (0.80) and our Zero-Shot from Unlabeled Data
# result which was also (0.80 - but with a slightly better positive review F1 score).
#
# As a final note - it should be mentioned that this sentiment analysis via generative "T5" text is a fair bit slower
# than the other methods we've looked at, but it DOES get very good results for unlabeled data, so that's the trade-off!
