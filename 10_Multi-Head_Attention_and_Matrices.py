import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec
import seaborn as sns
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity
from transformers import  AutoModelForCausalLM, AutoTokenizer

# Phi-3 models aren't keen on giving up their attention details so we'll use GPT2.
#
# Note: The original attention mechanism (which we're looking at here) is called "Multi-head Attention" – each head has
# its own set of key, query, and value projections. Newer variants include:
#   - "Multi-query Attention": uses a single key and value projection shared across all heads (reduces memory and improves speed).
#   - "Grouped-query Attention": uses fewer key and value projections than the number of heads by grouping heads to share them.
#
# Also: Attention is really about two main things:
#   - Scoring relevance between tokens, and
#   - Combining information.
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation="eager"  # We cannot use "flash_attention_2" when we want to see the attention vectors!
)

print(f"\n---Using model: {model_name} - which reports that it has:")
print(f"Model transformer layers: {model.config.n_layer}")
print(f"Model attention heads per layer: {model.config.n_head}")
print(f"Hidden size: {model.config.hidden_size}")

# Switch the model to EVALUATION mode so we do not drop values (from the `Dropout` layer, apparently) and "only use
# stored stats" in the `BatchNorm` layer. I'm not quite sure what this all means tbh, but that's why I'm learning this!
model.eval()

prompt = "The dog chased the squirrel because it"
print(f"\n--- Prompt is: {prompt}")

# Tokenise our prompt to get the token IDs...
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# ...then get what each token actually is.
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Generate outputs INCLUDING the attentions and hidden states!
# Note: Each attention head contains three projection matrices for queries, keys, and values.
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

# Access attention details & confirm the model config details are correct
# Note: The `numel()` counts the NUMber of ELements in an array or tensor! So data like [[1,1,1], [2,2,2]] would have a
# `numel` value of 6 as its shape is [2, 3] and 2 x 3 = 6.
attn_weights = outputs.attentions
num_attention_layers = len(attn_weights)
model_parameter_count = sum(p.numel() for p in model.parameters())
print(f"\n--- Confirming model: {model_name} has {num_attention_layers} attention layers and {model_parameter_count:,} parameters")

# Grab shape info from the first layer (we'll assume all layers have the same shape)
batch_size, num_heads, seq_len_q, seq_len_k = attn_weights[0].shape

print(f"\n--- Attention tensor shape info ---")
print(f"Batch size: {batch_size}")                # Will be 1 because we only provided 1 prompt
print(f"Number of attention heads: {num_heads}")  # GPT2 uses 12 layers for attention so has 12 attention heads
print(f"Sequence length (query): {seq_len_q}")    # The query matrix sequence length is 7 as that's the length of our prompt
print(f"Sequence length (key): {seq_len_k}")      # Same for our key matrix, it'll be 7 to match the length of our prompt

# Inspect a specific head's attention details.
#
# Note: Because our prompt has 7 tokens this is a 7x7 tensor - if our prompt was a different length these dimensions would change!
#
# Also: The relationship between the LM Head and each layer's Attention Heads is:
#
# LM Head	                                | Attention Heads
# ------------------------------------------------------------------------------------------
# Comes after all attention layers	        | Are within each transformer layer
# Maps final hidden states to vocabulary	| Map input tokens to context-aware representations
# Only one LM head	                        | Multiple attention heads per layer
#
# So they’re sequentially connected: attention heads → hidden states → LM head.
# The LM head consumes the output of all the attention processing, but it does not participate in attention itself.#
layer, head = 0, 0
print("\n---Attention weights (layer 0, head 0) are:")
print(attn_weights[layer][0, head])  # Shape: [seq_len, seq_len]

# Access hidden states for 'it' vs 'dog' comparison
hidden_states = outputs.hidden_states[-1][0]

# Grab the token indices from a word.
# Note: The GPT2 tokenizer uses Ġ for space, so if we try to look up `dog` we might fail if it's `Ġdog` (e.g., with a space before it)
def find_token_indices(source_tokens, target_word):
    possible_forms = [target_word, f"Ġ{target_word}"]
    return [i for i, tok in enumerate(source_tokens) if tok in possible_forms]

print("\n--- Tokens are: ---")
for t in tokens:
    print(t)

# Find the token IDs for `dog` and `it`
idx_dog = find_token_indices(tokens, "dog")
idx_it  = find_token_indices(tokens, "it")

# Then use those to get the embedding weights for those tokens IN THIS CONTEXT
# Note: We'll average the embedding values for the token in case we have multiple embeddings for `dog` or `it` (which
# would only occur if those tokens appeared multiple times in our token list - they DON'T with the current prompt, but
# if you change the prompt they might, so
emb_dog = hidden_states[idx_dog].mean(dim=0)
emb_it  = hidden_states[idx_it].mean(dim=0)

# Now get the similarity IN THIS CONTEXT between the embedding for `dog` and `it`
# Note: The `squeeze()` method removes dimensions of size 1 from an array, so emb_dog/it will go from size [1,768] to just [768]
similarity: Tensor = cosine_similarity(emb_dog.squeeze(), emb_it.squeeze(), dim=0)
print(f"\n--- Similarity between 'dog' and 'it' is: {similarity.item():.4f}")  # Should be 0.9599 - so very high correlation!

# ----- Visualize attention for layers and heads -----

# NOTE: I thought this visualisation of the attention details would show a high correlation between `dog` and `it - but
# IT DOES NOT! This is because attention is not an explanation, nor is it the final word on what the model "thinks"!
#
# Instead:
# - Attention tells us which tokens the model focuses on when processing another token!
#
# - When we see the high correlation between 'dog' and 'it, we're looking at the output of the last hidden state layer,
#   which has gone through:
#      - All 12 layers,
#      - All 12 attention heads per layer, and
#      - All MLP (Multi-Layer Perceptron) / Non-attention feedforward sub-layers!
#
# - Also, different attention heads can be used for different things, for example:
#      - Some are for co-referencing,
#      - Some are for syntactic dependency, and
#      - Some are for long-range attention!

batch_idx     = 0   # There's only a single batch because we only fed the model a single prompt
current_layer = 11  # Let's look at the last layer...
current_head  = 11  # ...and of that layer we'll look at the last attention head.

# Create figure with GridSpec to manage layout
fig = plt.figure(figsize=(8, 8))
gridspec = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[20, 1], width_ratios=[20, 1])
ax_heatmap = plt.subplot(gridspec[0, 0])
cbar_ax = plt.subplot(gridspec[0, 1])

plt.subplots_adjust(bottom=0.2)

def plot_attention_layer(layer_idx):
    ax_heatmap.clear()
    cbar_ax.clear()

    matrix = attn_weights[layer_idx][batch_idx, current_head].detach().cpu()

    sns.heatmap(
        matrix,
        cmap="viridis",
        ax=ax_heatmap,
        cbar=True,
        cbar_ax=cbar_ax,
        square=True,
        xticklabels=tokens,
        yticklabels=tokens,
    )

    ax_heatmap.set_title(f"Attention Weights - Layer {layer_idx} (Head {current_head})")
    ax_heatmap.set_xlabel("Key Tokens")
    ax_heatmap.set_ylabel("Query Tokens")

    # Overlay weights on the heatmap (optional)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            centered_x = x + 0.5
            centered_y = y + 0.5
            ax_heatmap.text(
                x=centered_x,
                y=centered_y,
                s=f"{matrix[x, y]:.2f}",
                ha='center',
                va='center',
                color='white',
                fontsize=10,
            )

    fig.canvas.draw_idle()

def next_layer(event):
    global current_layer
    current_layer = (current_layer + 1) % num_attention_layers
    plot_attention_layer(current_layer)

def prev_layer(event):
    global current_layer
    current_layer = (current_layer - 1) % num_attention_layers
    plot_attention_layer(current_layer)

def prev_head(event):
    global current_head
    current_head = (current_head - 1) % num_heads
    plot_attention_layer(current_layer)

def next_head(event):
    global current_head
    current_head = (current_head + 1) % num_heads
    plot_attention_layer(current_layer)

# Button locations
ax_prev_layer = plt.axes([0.15, 0.15, 0.15, 0.075])
ax_next_layer = plt.axes([0.32, 0.15, 0.15, 0.075])
ax_prev_head  = plt.axes([0.49, 0.15, 0.15, 0.075])
ax_next_head  = plt.axes([0.66, 0.15, 0.15, 0.075])

# Buttons
btn_prev_layer = Button(ax_prev_layer, 'Previous Layer')
btn_next_layer = Button(ax_next_layer, 'Next Layer')
btn_prev_head  = Button(ax_prev_head, 'Previous Head')
btn_next_head  = Button(ax_next_head, 'Next Head')

# OnClick handlers
btn_prev_layer.on_clicked(prev_layer)
btn_next_layer.on_clicked(next_layer)
btn_prev_head.on_clicked(prev_head)
btn_next_head.on_clicked(next_head)

# Initial draw
plot_attention_layer(current_layer)
plt.show()
