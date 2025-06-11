import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name: str = "microsoft/Phi-3-mini-4k-instruct"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
	pretrained_model_name_or_path=model_name,
	device_map="cuda",
	torch_dtype="auto",
	trust_remote_code=False,
	attn_implementation="eager"  # Skipping "flash_attention_2" as it takes forever to build
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Create a funny joke about computers.<|assistant|>"
print("--- Our prompt is: " + prompt)

# Tokenize the input prompt.
# This shows that the inputs that LLMs respond to are a series of INTEGERS! Each one is the unique ID for a specific
# token (character, word, or part of a word) - and each ID references a table inside the tokenizer containing all the
# tokens it knows about!
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
print("\n--- But the model doesn't see our text input! It sees:")
print(input_ids)

# But, we can DECODE the IDs back into text if we want to see how they map
print("\n--- Decoding these IDs back into the text for each token gives us:")
for input_id in input_ids[0]:
	decoded_input_id = tokenizer.decode(input_id)
	print(str(input_id) + "\t- " + decoded_input_id)

# Generate our response
generated_output = model.generate(
	input_ids=input_ids,
	max_new_tokens=25
)

# Print the output (as text)
print("\n--- The text of our output response is: ")
print(tokenizer.decode(generated_output[0]))

# But really, what we received was more integers which map to tokens!
print("\n--- But the model didn't respond with text - it responded with TOKENS!:")
for output_id in generated_output[0]:
	decoded_output_id = tokenizer.decode(output_id)
	print(str(output_id) + "\t- " + decoded_output_id)

# Note: Just as an example, the word "don't" is comprised as follows:
#   tensor(1016, device='cuda:0')	- don
#   tensor(29915, device='cuda:0')	- '
#   tensor(29873, device='cuda:0')	- t
# So really, it gets put together like this:
dont_tokens = [1016, 29915, 29873]
print(f"\n--- The word \"don't\" gets built from tokens 1016, 29915 & 29873 to become: {tokenizer.decode(dont_tokens)}")

# Clean up
gc.collect()

# Also: There are three major factors that dictate how a tokenizer breaks down an input prompt.
#
# First, at model design time, the creator of the model chooses a tokenization method. Popular methods include byte pair
# encoding (BPE) (widely used by GPT models) and WordPiece (used by BERT). These methods are similar in that they aim to
# optimize an efficient set of tokens to represent a text dataset, but they arrive at it in different ways.
#
# Second, after choosing the method, we need to make a number of tokenizer design choices like vocabulary size and what
# special tokens to use.
#
# Third, the tokenizer needs to be trained on a specific dataset to establish the best vocabulary it can use to
# represent that dataset. Even if we set the same methods and parameters, a tokenizer trained on an English text dataset
# will be different from another trained on a code dataset or a multilingual text dataset.
#
# Some different tokenizers are:
# BERT (uncased) [2018] - Vocab size: 30,522
#   - Type: WordPiece
#   - Cannot handle newlines, all text is lowercase, cannot handle emojis
#
# BERT (cased) [2018] - Vocab size: 30,522
#   - Type: WordPiece
#   - Cannot handle newlines, text can be UPPER or lower case, cannot handle emojis
#
# GPT-2 [2019] - Vocab size: 50,257
#   - Type: Byte Pair Encoding (BPE)
#   - Handles newlines, capitalisation, emojis (as multiple tokens), and tabs
#
# Flan-T5 [2022] - Vocab size: 32,100
#   - Type: SentencePiece
#   - Cannot handle newlines or emojis but can do upper/low
#
# GPT-4 [2023] - Vocab size: ~100K
#   - Type: Byte Pair Encoding (BPE)
#   - Handles newlines, upper/lower case, emojis, has special tokens for programming terms ('elif' etc.)
#   - Note: Has special tokens for all whitespace amounts up to 83 spaces!
#
# StarCoder2 [2024] - Vocab size: 49,152
#   - Type: Byte Pair Encoding (BPE)
#   - Handles everything GPT-4 does, designed for code, has special tokens used to reference specific repos and files within them.
#
#  etc. etc.
