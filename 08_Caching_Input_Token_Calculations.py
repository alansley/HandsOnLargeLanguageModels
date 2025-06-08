import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name: str = "microsoft/Phi-3-mini-4k-instruct"

# Load our model
model = AutoModelForCausalLM.from_pretrained(
	pretrained_model_name_or_path=model_name,
	device_map="cuda",
	torch_dtype="auto",
	trust_remote_code=False,                 # Note: This must be False for Phi-3 models as they're a bit old
	attn_implementation="flash_attention_2"  # Can just use "eager" if we don't want to use flash attention
)

# Create an appropriate tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Warm up the pipeline by just giving it some stupid busywork - this way the setup duration doesn't count towards our
# Cached Vs. Non-Cached timings. Side effect: Creates an awesome love song =P
warmup_prompt = "Write me a love song about a bacon sandwich"
song_input_ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to("cuda")
song_output_ids = model.generate(
	input_ids=song_input_ids,
	max_new_tokens=200
)
the_greatest_love_song_in_the_world_about_a_bacon_sandwich = tokenizer.decode(song_output_ids[0], skip_special_tokens=True)

prompt = "Write a very long email apologizing to Princess Peach for the tragic gardening mishap. Explain how it happened."
# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

# Function to time the execution of the provided function and return the generated text plus duration
def time_execution_of(some_function) -> tuple[str, float]:
	start_time = time.time()
	result = some_function()
	duration = time.time() - start_time
	return result, duration

max_tokens = 1000

# A lambda that generates some text and uses caching
def generate_text_with_cache() -> str:
	output_ids = model.generate(
		input_ids=input_ids,
		max_new_tokens=max_tokens,
		use_cache=True  # The default is to use key/value caching - but we'll be specific about it here
	)
	return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# A lambda that generates some text where caching is explicitly disabled
def generate_text_without_cache() -> str:
	output_ids = model.generate(
		input_ids=input_ids,
		max_new_tokens=max_tokens,
		use_cache=False  # Without key/value caching we'll process every single input token right the way through all the transformer blocks!
	)
	return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Do the text generation and print out the timings (we'll print the output length too, just to make sure they equiv.).
# Note: On my RTX 4090 I get about ~28 seconds for the caching run and ~45 seconds for the non-caching run w/ 1000 tokens.
# Also: Caching generation takes around ~120W, while non-caching can hit ~180W and introduces some coil-whine on my setup!
caching_text, duration_with_caching = time_execution_of(generate_text_with_cache)
print(f"Generation time WITH    cache: {duration_with_caching:.4f} seconds. Caching text length: {len(caching_text)}.")
noncaching_text, duration_without_caching = time_execution_of(generate_text_without_cache)
print(f"Generation time WITHOUT cache: {duration_without_caching:.4f} seconds. Non-caching text length: {len(noncaching_text)}.")

def remove_blank_lines(text: str) -> str:
	return "\n".join(line for line in text.splitlines() if line.strip())

# Print the text we generated, just so we know & can prove we actually did some work!
print_char_count = 300
print(f"\nCache text is: {remove_blank_lines(caching_text)[:print_char_count]}")
print("=" * 100)
print(f"\nNon-cache text is: {remove_blank_lines(noncaching_text)[:print_char_count]}")

# Just for lolz
print("=" * 100)
print(the_greatest_love_song_in_the_world_about_a_bacon_sandwich)
