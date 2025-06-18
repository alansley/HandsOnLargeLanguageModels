# Note: This code is adapted from that on p133 which accesses OpenAI and requires an API key, but I don't re

from openai import OpenAI
import os

# IMPORTANT: You need an OpenAI key and credits to run this example - put the key as a single line in the given file.
#
# ALSO: If you haven't bought OpenAI credits and are not within 3 months of creating your OpenAI account then you will
# need to PURCHASE credits to test with because your free $5 worth of credits will have expired!
#
# See: https://community.openai.com/t/are-openai-credits-expiring/511215
#
# I guess you can just create another free account, but still - it's money-grubbing and a bad look. Also keep in mind
# that even purchased credits expire after 1 year, because money grubbing.
#
# CAREFUL: This file is excluded from git via the .gitignore included with this repo - if you replace or adjust that
# .gitignore and aren't careful you could end up with your API key on your repo. Don't do that.
OPENAI_API_KEY_FILE = "DataFiles/openai_api_key.txt"

if not os.path.exists(OPENAI_API_KEY_FILE):
    print(f"No OpenAI API key file at {OPENAI_API_KEY_FILE} - bailing!")
    quit(-1)

with open(OPENAI_API_KEY_FILE, "r") as api_file:
    openai_api_key = api_file.readline().strip()  # If we have a newline character at the end it doesn't work - so strip!

client = OpenAI(api_key=openai_api_key)

raw_prompt = """Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any
other answers.
"""

review = "The film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game."

substituted_prompt = raw_prompt.replace("[DOCUMENT]", review)

# Currently the cheapest option at 40c US per 1 million input tokens and $1.60 US per 1 million output tokens. This will
# change - so likely best to see: https://platform.openai.com/docs/pricing
model_name = "gpt-4.1-mini"

response = client.responses.create(
    model=model_name,
    input=substituted_prompt
)

# Make sure we got the 0 (negative) or 1 (positive) we were expecting
if response.output_text != "0" and response.output_text != "1":
    print(f"Model wasn't listening to our prompt properly - we expects 0/1 but got: {response.ouput_text}")
    quit(-2)

# Show our result
pos_or_neg = "negative"
if response.output_text == "1":
    pos_or_neg = "positive"
print(f"Based on the review:\n{review},\n\nThe model {model_name} thinks this review is: {pos_or_neg}")

# Note: If you want to test if any errors are coming from our Python code or not you can test if things are working with
# tha API key by slapping the following into the terminal, replacing <YOUR_OPEN_AI_KEY_HERE> with YOUR key (don't keep
# the angle brackets either):
#
# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer <YOUR_OPENAI_API_KEY_HERE>" \
#   -d '{
#     "model": "gpt-4o-mini",
#     "store": true,
#     "messages": [
#       {"role": "user", "content": "write a haiku about openai not being open at all"}
#     ]
#   }'
