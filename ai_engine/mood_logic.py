import random
import json
import os
from datetime import datetime

MEMORY_PATH = os.path.join(os.path.dirname(__file__), '..', 'memory', 'mood_state.json')

def load_mood():
    with open(MEMORY_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def update_mood(new_mood):
    data = load_mood()
    data['current_mood'] = new_mood
    data['mood_history'].append({
        "timestamp": datetime.now().isoformat(),
        "mood": new_mood
    })
    with open(MEMORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def get_current_mood():
    data = load_mood()
    return data.get("current_mood", "neutral")

def adjust_mood_by_input(user_input):
    mood_map = {
        "happy": ["great", "awesome", "love", "fantastic"],
        "sad": ["tired", "bored", "sad", "depressed"],
        "romantic": ["kiss", "hug", "romantic", "touch"],
        "angry": ["angry", "mad", "furious"],
        "anxious": ["scared", "worried", "nervous"]
    }

    for mood, keywords in mood_map.items():
        if any(word in user_input.lower() for word in keywords):
            update_mood(mood)
            return mood

    return "neutral
