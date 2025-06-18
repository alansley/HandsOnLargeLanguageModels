import gc
from torch import Tensor, softmax, no_grad, cuda, topk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.nn.functional import cosine_similarity

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

prompt = "The dog chased the squirrel because it"

# Tokenize and move to GPU
inputs: dict[str, Tensor] = tokenizer(prompt, return_tensors="pt").to("cuda")

# Get model output logits without generating more tokens.
# Note: `with no_grad()` we're disabling gradient tracking within the scope of the called commands, this is shorthand for:
#
#   ctx = torch.no_grad()
#   ctx.__enter__()
#   try:
#         outputs = model(**inputs, output_hidden_states=True)
#   finally:
#         ctx.__exit__(None, None, None)
#
with no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Hidden states is a tuple: (layer_0, layer_1, ..., layer_n)
# Each hidden state tensor has shape: [batch_size, seq_len, hidden_dim]
# We'll use the **last layer** hidden state
hidden_states = outputs.hidden_states[-1][0]  # shape: [seq_len, hidden_dim]

# Get the logits for the **next token**
logits: Tensor = outputs.logits     # shape: [batch, sequence_length, vocab_size]
next_token_logits = logits[0, -1]   # Get the logits for the last token

# Convert logits to probabilities
probs = softmax(next_token_logits, dim=0)

# Show top however many predicted tokens
top_k = 5
top_probs, top_indices = topk(probs, top_k)

print(f"\n--- What are the most likely next tokens in the prompt: {prompt}")
print(f"{'Rank':<5} {'Token ID':<10} {'Prob (%)':<10} {'Token'}")
print("-" * 50)
for i in range(top_k):
    token_id = top_indices[i].item()
    prob = top_probs[i].item() * 100
    token_str = repr(tokenizer.decode([token_id]))  # Use repr to show special chars
    print(f"{i+1:<5} {token_id:<10} {prob:<10.2f} {token_str}")

# --- Okay, now let's find out if "it" is more likely to mean the dog or the squirrel...

# Map token IDs to string tokens for context
input_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# Debug print: Print out the list of tokens in the prompt
print("\n--- Token breakdown is:")
for i, tok in enumerate(tokens):
    print(f"{i:2}: {tok}")

# Indices based on your token printout
idx_dog = 1               # 'dog' is a single token...
idx_squirrel = [5, 6, 7]  # ...but 'squirrel' is made up from tokens [5,6,7] ("squ", "ir", and "rel")...
idx_it = 9                # ...and 'it' is a single token.

# Get embeddings
emb_dog      = hidden_states[idx_dog]
emb_squirrel = hidden_states[idx_squirrel].mean(dim=0)  # As "squirrel" is 3 tokens we have to average their embedding values!
emb_it       = hidden_states[idx_it]

# Now use cosine similarities to compare the embedding value of 'it' to both 'dog' and 'squirrel'
sim_dog      = cosine_similarity(emb_it, emb_dog, dim=0).item()
sim_squirrel = cosine_similarity(emb_it, emb_squirrel, dim=0).item()

print(f"\nCosine similarity between 'it' and 'dog' in this context: {sim_dog:.4f}")
print(f"Cosine similarity between 'it' and 'squirrel' in this context: {sim_squirrel:.4f}")
print("As such, 'it' is more likely referring to:", "dog" if sim_dog > sim_squirrel else "squirrel")

# Clean up
cuda.empty_cache()
gc.collect()
