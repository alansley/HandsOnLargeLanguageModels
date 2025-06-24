# === 🔥 CHAOS ENGINE: Fully Randomized Text Gen Sweep ===
# ChatGTP Evaluation 2025/06/24

import time
import torch
import random
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ----------------------------
# 📌 Config
# ----------------------------

PROMPT = "Quite an experience to live in fear, isn't it? That's what it is to be a slave."
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 60
NUM_RETURN_SEQUENCES = 1

top_k_values = [5, 50, 200]
top_p_values = [0.5, 0.9, 0.95]
temperatures = [0.1, 0.7, 1.2]

# Okay...
if torch.cuda.is_available():
    print(f"🔥 Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA not available – using CPU.")

# ----------------------------
# 🚀 Load tokenizer once
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ----------------------------
# 🔧 Generator with runtime-random each time
# ----------------------------

def get_generator():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# ----------------------------
# 🔁 Generation Call
# ----------------------------

# def call_generator(prompt, temperature, top_k, top_p):
#     # 🔀 Re-seed at runtime with time-based randomness
#     seed = int(time.time() * 1000) % (2**32 - 1)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = True
#
#     # ♻️ Reload model + pipeline every run to prevent any cache reuse
#     generator = get_generator()
#     generator.model.generation_config = GenerationConfig()
#
#     kwargs = {
#         "max_new_tokens": MAX_NEW_TOKENS,
#         "num_return_sequences": NUM_RETURN_SEQUENCES,
#         "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
#     }
#
#     if temperature > 0:
#         kwargs.update({
#             "do_sample": True,
#             "temperature": temperature,
#             "top_k": top_k,
#             "top_p": top_p,
#             "repetition_penalty": 1.2,
#         })
#     else:
#         kwargs["do_sample"] = False
#
#     return generator(prompt, **kwargs)


def call_generator(prompt, temperature, top_k, top_p):
    import time
    import random

    # 🔀 Re-seed at runtime with time-based randomness
    seed = int(time.time() * 1000) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # ♻️ Reload model + pipeline every run to prevent any cache reuse
    generator = get_generator()
    generator.model.generation_config = GenerationConfig()

    kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_return_sequences": NUM_RETURN_SEQUENCES,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        # "min_new_tokens": 10,  # Optional: forces a minimum output
    }

    if temperature > 0:
        kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": 1.2,
        })
    else:
        kwargs["do_sample"] = False

    clean_prompt = prompt.rstrip() + "\n"
    outputs = generator(clean_prompt, **kwargs)

    # 🧪 Detect and warn on null/no continuation
    generated_text = outputs[0].get("generated_text", "").strip()
    if not generated_text or generated_text == clean_prompt.strip():
        print("⚠️ Model returned no continuation – likely due to sampling limits or decoding failure.\n")
        return "[No response generated]"

    if generated_text.startswith(clean_prompt):
        return generated_text[len(clean_prompt):].lstrip()
    else:
        return generated_text


# ----------------------------
# 📌 Description helpers
# ----------------------------

def describe_settings(k, p, t):
    k_desc = "focused" if k <= 5 else "creative" if k <= 50 else "chaotic"
    p_desc = "constrained" if p <= 0.5 else "flexible" if p <= 0.9 else "unfiltered"
    t_desc = "precise" if t <= 0.3 else "balanced" if t <= 0.8 else "wild"
    return f"🧠 Expect {k_desc}, {p_desc}, {t_desc} output"

# ----------------------------
# 🔁 Full sweep
# ----------------------------

print("********************************************************************************************************************")
print("CAREFUL: It's possible for this script to print some DERANGED SHIT - don't take it personally, it's just word-salad.")
print("********************************************************************************************************************\n")

# for tk in top_k_values:
#     for tp in top_p_values:
#         for temp in temperatures:
#             print(describe_settings(tk, tp, temp))
#             print(f"=== top_k = {tk}, top_p = {tp}, temperature = {temp} ===")
#
#             prompt = PROMPT.rstrip() + "\n"
#             outputs = call_generator(prompt, temp, tk, tp)
#             text = outputs[0]["generated_text"]
#
#             response_only = text[len(prompt):].lstrip() if text.startswith(prompt) else text
#
#             print(f"quote   : {PROMPT.strip()}")
#             print(f"response: {response_only}\n")


for tk in top_k_values:
    for tp in top_p_values:
        for temp in temperatures:
            print(describe_settings(tk, tp, temp))
            print(f"=== top_k = {tk}, top_p = {tp}, temperature = {temp} ===")

            prompt = PROMPT.rstrip() + "\n"
            generated_text = call_generator(prompt, temp, tk, tp)

            response_only = generated_text[len(prompt):].lstrip() if generated_text.startswith(prompt) else generated_text

            print(f"quote   : {PROMPT.strip()}")
            print(f"response: {response_only}\n")
