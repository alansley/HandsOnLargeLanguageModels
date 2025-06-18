import gc
from transformers import AutoModel, AutoTokenizer

# Load a language model - in this case we'll pick one that can work with the DeBERTa tokenizer
# Note: "BERT" standards for "Bidirectional Encoder Representations from Transformers"
# Further reading: https://en.wikipedia.org/wiki/BERT_(language_model)
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

# Load a tokenizer
# DeBERTa v3 is a small and efficient tokenizer described in the following paper:
# “DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training gradient-disentangled embedding sharing”.
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Tokenize some input
input_txt = "Hello World"
tokens = tokenizer(input_txt, return_tensors='pt')

# Process the tokens & print the output
# Note: This will be an array of tensor values, not token IDs!
# We get output like:
#
# --- The shape of the output is: torch.Size([1, 4, 384])
# --- Values are:
# tensor([[[-3.2520,  0.1818, -0.1254,  ..., -0.0502, -0.2334,  0.8897],
#          [-0.4673,  0.1730, -0.0206,  ..., -0.5289,  0.7303,  2.1177],
#          [-0.4950,  0.0564,  0.2842,  ...,  1.0543, -0.1747,  1.3793],
#          [-2.9601,  0.2129, -0.1138,  ...,  0.1518, -0.2094,  1.0494]]],
#        grad_fn=<NativeLayerNormBackward0>)
#
# So looking at the shape of our tokenized values for our input "Hello World" we get: torch.Size([1, 4, 384])
#   - The first array of size 1 is just the BATCH SIZE (we gave it 1 sentence as input). We can ignore this for now as
#   this is mainly used when TRAINING the tokenizer or sending it multiple inputs at the same time to speed things up),
#   - Then we can read the "4, 384" part as:
#       - Our input was broken up into 4 tokens,
#       - Where each one contains a vector of 384 values - so essentially these 384 values (the "contextualized
#       embedding" indicate the MEANING of the token in this context! For example, the word "Hello" in "Hello
#       Friend" has a different meaning to the same word "Hello" in "Say Hello to my little friend!" - because the
#       CONTEXT of the token is different!
#       - Another way to think of this is that the 384 (in this case) values represent the token's meaning IN CONTEXT in
#       384 dimensional space!
tokenized_output = model(**tokens)[0]
print("\n--- The shape of the tokenized output is: " + str(tokenized_output.shape))
print("\n--- Values are:")
print(tokenized_output)

# Decode & print the output.
# We can see where we got the 4 tokens from in the above embedding output now, because for "Hello World" - we get:
#
# [CLS]    <-- 1 - "[CLS]" just marks the beginning of a string or sentence
# Hello    <-- 2
#  World   <-- 3 - Notice the space before the word "World" - so this means " World" is its own separate token! Almost - see below - it's a `Ġ`!
# [SEP]    <-- 4 - "[SEP]" just marks the end of a string or sentence
#
print("\n--- Decoded tokenized output is:")
for token in tokens['input_ids'][0]:
	decoded_token = tokenizer.decode(token)
	print(decoded_token)

# ChatGPT says to properly show the tokenized output I should use:
# Convert input IDs back to token strings (shows spacing correctly)
# What we'll see from this is:
#
# [CLS]
# Hello
# ĠWorld    <-- Notice how there isn't a SPACE before "World" now, it's a `Ġ` character, which indicates a leading space!
# [SEP]
print("\n--- Tokens with actual symbol embedding (look at the \"World\" line):")
tokens_as_strings = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
for token in tokens_as_strings:
	print(token)

# ----- Just for fun, we'll send two messages to the tokenizer at once -----

# Two input sentences
multiple_inputs = ["Hello World", "Goodbye Moon"]

# Tokenize both at once (batch encoding)
tokens = tokenizer(multiple_inputs, return_tensors='pt', padding=True)  #, truncation=True)

# Get model output (contextual embeddings)
# Note: The "last hidden state" is the final vector embedding (i.e., the 384 values that define the meaning IN CONTEXT
# for that token) at the FINAL layer of the token. In fact, the token has different vector values at each layer, which
# get increasingly refined as we pass through the different layers of the tokenizer/transformer via the various
# attention mechanisms and feed-forward networks.
output = model(**tokens)
last_hidden_states = output.last_hidden_state

# Print shape: should be [batch_size, max_seq_len, hidden_size]
print(f"\nWhen providing multiple (2 in this case) inputs, the shape of the output is: {last_hidden_states.shape}")
# e.g. torch.Size([2, 5, 384]) if each sentence ends up with 5 tokens

# Show which tokens were created
# Output:
# Decoded tokens per input:
# Input 1: ['[CLS]', 'Hello', 'ĠWorld', '[SEP]', '[PAD]']       <-- The [PAD] token here ensures each array is of the same size (5) in this example
# Input 2: ['[CLS]', 'Good',  'bye',    'ĠMoon', '[SEP]']
print("\nDecoded tokens per input (note the padding token at the end of Input 1 so we don't have a jagged array!):")
for i, input_ids in enumerate(tokens['input_ids']):
	token_strs = tokenizer.convert_ids_to_tokens(input_ids)
	print(f"Input {i + 1}: {token_strs}")

# ----- If we really want to, we can get NON-FINAL LAYER contextual embeddings -----

# Tet's get the tokens for "Hello World" again
tokens = tokenizer(input_txt, return_tensors='pt')

# Now let's transform them into the embeddings and KEEP the intermediate "hidden" states in our output
output = model(**tokens, output_hidden_states=True)

# This is a tuple of [embedding, layer1, ..., final layer]
# NOTE: We can think of layer 0 as, essentially, "Here is the dictionary definition of the word/token" (created when
# training the tokeniser) - and then each subsequent layer is used to increasingly move towards "Here is the definition
# of the word/token IN CONTEXT OF THIS SENTENCE"!
hidden_states = output.hidden_states
print("The total number of layers in our tokenizer/transformer are: " + str(len(hidden_states)))

# Get the second-to-last layer:
penultimate_layer_output = hidden_states[-2]

print("\n--- Looking at the PENULTIMATE layer, the shape of the tokenized output is: " + str(penultimate_layer_output.shape))
print("\n--- Values in the PENULTIMATE layer are:")
print("(Compare these to the original outputs above for \"Hello World\" - there aren't even close to being the same!)")
print(penultimate_layer_output)

# Clean up
gc.collect()
