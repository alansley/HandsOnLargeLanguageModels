import torch
from torch import Tensor, softmax, no_grad
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
	do_sample=False,          # If this is False we'll just pick the most probable next token for output, which will result in the same output every run!
	#temperature=0.9          # Temperature will be applied (0..1 -> min..max creativity) ONLY if `do_sample` is True - if it's False this has no effect!
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
#         outputs = model(**inputs)
#   finally:
#         ctx.__exit__(None, None, None)
#
# Because Python's gonna Python...
with no_grad():
    outputs = model(**inputs)

# Get the logits for the **next token**
logits: torch.Tensor = outputs.logits  # shape: [batch, sequence_length, vocab_size]
next_token_logits = logits[0, -1]  # Get the logits for the last token

# Convert logits to probabilities
probs = softmax(next_token_logits, dim=0)

# Show top 20 predicted tokens
top_k = 5
top_probs, top_indices = torch.topk(probs, top_k)

print(f"\nPrompt: {prompt!r}\n")
print(f"{'Rank':<5} {'Token ID':<10} {'Prob (%)':<10} {'Token'}")
print("-" * 50)
for i in range(top_k):
    token_id = top_indices[i].item()
    prob = top_probs[i].item() * 100
    token_str = repr(tokenizer.decode([token_id]))  # Use repr to show special chars
    print(f"{i+1:<5} {token_id:<10} {prob:<10.2f} {token_str}")

def get_log_prob_of_continuation(text):
    #inputs = tokenizer(text, return_tensors="pt").to("cuda")
    #with torch.no_grad():
    #    outputs = model(**inputs)
    #logits = outputs.logits

    # Calculate log probabilities of the last generated token
    # We want to score the last token only (to see likelihood of that word after the prefix)
    #last_token_id = inputs.input_ids[0, -1]
    last_token_id = inputs["input_ids"][0, -1]
    last_token_logits = logits[0, -2]  # logits at second last position correspond to next token
    log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
    return log_probs[last_token_id].item()

text1 = "The dog chased the squirrel because the dog was"
text2 = "The dog chased the squirrel because the squirrel was"

log_prob1 = get_log_prob_of_continuation(text1)
log_prob2 = get_log_prob_of_continuation(text2)

print(f"Log prob for 'dog was': {log_prob1}")
print(f"Log prob for 'squirrel was': {log_prob2}")