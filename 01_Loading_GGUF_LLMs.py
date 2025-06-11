# Note: We need the `gguf` package to load GGUFs - but they're super slow to load so we'll stick with SafeTensors or
# such for the rest of the code.
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# To use GGUF format models we need to specify the model and the quant
model_id       = "bartowski/Phi-3.1-mini-4k-instruct-GGUF"
model_filename = "Phi-3.1-mini-4k-instruct-Q6_K_L.gguf"

# Load our model
# Note: We cannot use "flash_attention" for the attn_implementation argument here!
model = AutoModelForCausalLM.from_pretrained(
	pretrained_model_name_or_path=model_id,
	gguf_file=model_filename,
	device_map="cuda",
	torch_dtype="auto",
	trust_remote_code=False,     # Note: This needs to be false for Phi-3 models as they're a bit old
	attn_implementation="eager"  # Also: "flash_attention_2" doesn't work with this particular model
)

# Print some details regarding how it'll generate responses
print(model.generation_config)

# Create an appropriate tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=model_filename)

# Create a pipeline so we can talk to the model
generator = pipeline(
	task="text-generation",
	model=model,
	tokenizer=tokenizer,
	return_full_text=False,  # If true the model will return the prompt it was given PLUS the response
	max_new_tokens=500,      # Limit to a short response
	do_sample=True,          # If this is False we'll just pick the most probable next token for output, which will result in the same output every run!
	temperature=0.9          # Temperature will be applied (0..1 -> min..max creativity) ONLY if `do_sample` is True - if it's False this has no effect!
)

# Create a prompt to send to the LLM
# The prompt (user input / query)
ROLE_FIELD_NAME: str = "role"
USER_ROLE: str = "user"
CONTENT_FIELD_NAME: str = "content"
GENERATED_TEXT_FIELD_NAME: str = "generated_text"
messages = [
	{
		ROLE_FIELD_NAME: USER_ROLE,
		CONTENT_FIELD_NAME: "Create a funny joke about chickens."
	}
]

# Generate and print the output!
output = generator(messages)
print(output[0][GENERATED_TEXT_FIELD_NAME])

# Clean up
gc.collect()
