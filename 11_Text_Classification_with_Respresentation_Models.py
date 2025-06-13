import os
from datasets import load_dataset

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
