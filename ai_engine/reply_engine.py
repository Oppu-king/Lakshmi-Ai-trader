import json
import random
import os

def load_responses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_reply(user_input, mood="neutral"):
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    files = [
        "general_responses.json",
        "emotional_responses.json",
        "romantic_responses.json",
        "trading_responses.json"
    ]

    for filename in files:
        responses = load_responses(os.path.join(base_dir, filename))
        for pattern, replies in responses.items():
            if pattern.lower() in user_input.lower():
                return random.choice(replies)

    return "Tell me more, love… I'm here to listen."
