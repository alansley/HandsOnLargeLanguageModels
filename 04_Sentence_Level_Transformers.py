# Rather than have a token embedding that references just a single word or part of a word, there exist tokenizers that
# can represent sentences or even whole documents! For example, we might have a vector/embedding for the sentence
# "Say the line, Bart!" or the even the entire Declaration of Independence (or the lyrics to a song etc.)!

import gc

# Note: We need to run `pip install sentence_transformers` before we can use this!
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

sentence = "Best movie ever!"

# Encode the entire sentence "Best movie ever!" as a single vector
vector = model.encode(sentence)

print(f"The size of the vector for the sentence \"{sentence}\" is: {str(vector.shape)}")  # Should be (768,) for this model

# The values themselves are just that - we'll print out the first dozen as an example
print("\n--- Here's the first dozen vectors in the encoding:")
print(vector[:12])

# Clean up
gc.collect()
