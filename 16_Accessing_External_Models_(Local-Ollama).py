# In this example we'll connect to a locally running instance of Ollama, run one via "ollama run qwen3:0.6b" [523MB] or
# similar. Model card: https://ollama.com/library/qwen3
#
# Note: I don't recomment using "phi3" for this - it doesn't listen to the prompt properly and keeps talking about why
# it thinks what it thinks rather than just returning a 0 or 1.
from typing import Any

import requests
from requests import Response

LOCAL_OLLAMA_PORT = 11434  # Default

OLLAMA_API_URL = f"http://localhost:{LOCAL_OLLAMA_PORT}/api/generate"
MODEL = "qwen3:0.6b"

def classify_sentiment_with_ollama(document: str, print_json: str = True) -> str | None:
    prompt = f"""Predict whether the following document is a positive or negative movie review:

{document}

If it is positive return 1 and if it is negative return 0. Do not give any other answers - just a single 0 or a 1."""

    http_response: Response = requests.post(OLLAMA_API_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })

    if http_response.status_code != 200:
        print(f"Request failed: {http_response.status_code} - {http_response.text}")
        return None

    result_json = http_response.json()
    output = result_json.get("response", "").strip()

    if print_json:
        print(f"Raw JSON response:\n{result_json}\n")

    return output


# Pass our review to ollama & get its response
review = "The film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game."
response: str = classify_sentiment_with_ollama(review).strip()

def print_response_details():
    global response

    # If we used a reasoning model then remove everything inside the <think>...</think> tags
    print_reasoning = True
    if response.startswith("<think>"):
        close_thinking_index = response.find("</think>")
        if close_thinking_index != -1:
            reasoning_details = response[:close_thinking_index + 8]
            if print_reasoning:
                print(f"Reasoning details:\n{reasoning_details}")

            # Update the response to be only the last character (should be 0 or 1)
            response = response[-1]
        else:
            print("Somehow found opening <think> tag but couldn't find closing </think>!")
            quit(-1)

    print(f"\nReview:\n{review}")

    # Print our result
    if len(response) == 1:
        if response == "1":
            response_string: str = "Positive"
        else:
            response_string: str = "Negative"
        print(f"\nOllama ({MODEL}) thinks this review is: {response_string}\n\n")
    else:
        print(f"Somehow don't get a single character response - we got: {response}")

# For our positive review
print_response_details()

# Bonus negative review
review = "Maybe don't name your musical 'Rent' if you don't even have a single song about leasing law, property management procedures, or net lease calculations. As a real estate professional I am very disappointed and feel I was misled."
response = classify_sentiment_with_ollama(review).strip()
print_response_details()

# IDK, just playing with funny reviews now...
review = "It's this movie where Colin Farrell is in a phone booth, and someone calls the phone booth, and if he leaves the phone booth he'll get shot. I think it was called 'The phone that couldn't hang up'"
response = classify_sentiment_with_ollama(review).strip()
print_response_details()
