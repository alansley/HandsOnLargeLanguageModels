import gc
from torch import Tensor, topk, cuda
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name: str = "microsoft/Phi-3-mini-4k-instruct"

# Load our model
model = AutoModelForCausalLM.from_pretrained(
	pretrained_model_name_or_path=model_name,
	device_map="cuda",
	torch_dtype="auto",
	trust_remote_code=False,                 # Note: This NEEDS to be false for Phi-3 models as they're a bit old
	attn_implementation="flash_attention_2"  # Can just use "eager" if we don't want to use flash attention
)

# Create an appropriate tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a pipeline so we can talk to the model
generator = pipeline(
	task="text-generation",
	model=model,
	tokenizer=tokenizer,
	return_full_text=False,  # If true the model will return the prompt it was given PLUS the response
	max_new_tokens=50,       # Limit to a short response
	do_sample=False,         # If this is False we'll just pick the most probable next token for output, which will result in the same output every run!
	#temperature=0.9         # Temperature will be applied (0..1 -> min..max creativity) ONLY if `do_sample` is True - if it's False this has no effect!
)

# Print some details about the model
print(model)

# Printing as above will get something like the following:
# Note: 90% of it is talking about the `model` (32 layers) then we have a final LM Head
#
# Phi3ForCausalLM(
#   (model): Phi3Model(                                         <-- All this block downwards in the model
#     (embed_tokens): Embedding(32064, 3072, padding_idx=32000) <-- This model has ~32K tokens, where each token has a ~3K vector of embeddings!
#     (layers): ModuleList(
#       (0-31): 32 x Phi3DecoderLayer(                          <-- There are 32 layers (i.e., transformer blocks)
#         (self_attn): Phi3Attention(                           <-- Each block has an attention layer
#           (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
#           (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
#         )
#         (mlp): Phi3MLP(                                       <-- Each block also has a feed-forward layer aka "Multi-Level Perceptron"
#           (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
#           (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
#           (activation_fn): SiLU()
#         )
#         (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
#         (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
#         (resid_attn_dropout): Dropout(p=0.0, inplace=False)
#         (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
#       )
#     )
#     (norm): Phi3RMSNorm((3072,), eps=1e-05)
#     (rotary_emb): Phi3RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=3072, out_features=32064, bias=False)      <-- ...then the LM head chooses the final token out of the ~32K it knows!
# )

prompt = "The capital of France is"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Run the model to get hidden states (before lm_head)
# Note: If we have 5 tokens in "The capital of France is" - when we perform this modelling step we actually process
# EACH AND EVERY TOKEN right the way through from its initial embedding value down to the final output vector that comes
# before the LM Head! However, the LM Head output ONLY uses the output vector from the FINAL token to predict the output
# token (the capital of France being Paris, in this instance). So why process EVERY token?
#
# From p82: "The answer is that the final calculations of the previous streams are required and used in calculating
# the FINAL stream" (that goes into the LM Head).
model_output = model.model(input_ids)
print(f"\nModel output shape is: {model_output[0].shape}")

# Pass the hidden states through the LM head to get "logits".
# Note: "logits" are the probability SCORES - but they are not the ACTUAL probabilities like "70% chance" or something.
# Instead, they're just a score like -1.2 or 33.7. To convert a logit to a probability we must apply a `softmax`
# processing step to the logit.
lm_head_output: Tensor = model.lm_head(model_output[0])

# The shape will be [1, sequence_length, vocab_size], e.g. [1, 5, 32064], where:
#    - 1 is the batch size,
#    - sequence_length is how many tokens the prompt was broken into (remember that with some tokenisers words can be
#      broken up into multiple tokens!)
#    - vocab_size is the number of possible tokens in the tokeniser's vocabulary (i.e., all the tokens it knows about
#      and can use!).
print(f"\nLM Head shape is: {lm_head_output.shape}")

# Get the logits (unnormalized scores) for the last token in the sequence
last_tensor = lm_head_output[0, -1]  # Batch 0, last token. Shape will be [32064]

# Now get the INDEX of the token with the highest score
# IMPORTANT: `argmax` gets the index of the largest value, not the actual value itself like `max` would!
# NOTE: We _could_ get the most likely next token ID via a single line like:
#         `token_id = lm_head_output[0,-1].argmax(-1)`
#       But I've broken it up into steps for simplicity.
token_id = last_tensor.argmax()
print(f"\nThe most probably next token has ID: {token_id}")

# Decode the token ID into an actual string
token_string: str = tokenizer.decode(token_id)

print(f"\nFrom our prompt: {prompt}")
print(f"We get the response: {token_string}")

# Just for fun, we'll take a look at the next few candidate tokens which we might have chosen
next_most_likely_count = 4

# Get the top `next_most_likely_count` logits and their token IDs
top_logits, top_token_ids = topk(last_tensor, k=next_most_likely_count)

# Decode those token IDs into strings
top_tokens = tokenizer.batch_decode(top_token_ids)

# Print each token and its associated logit
print(f"\n{'Rank':<5} | {'Token ID':<10} | {'Logit':<10} | Token")
print("-" * 54)
for i in range(next_most_likely_count):
    rank = i + 1
    token_id = top_token_ids[i].item()
    logit = top_logits[i].item()
    token = repr(top_tokens[i])  # Use repr() to show whitespace and special characters clearly
    print(f"{rank:<5} | {token_id:<10} | {logit:<10.4f} | {token}")

# Clean up
cuda.empty_cache()
gc.collect()
