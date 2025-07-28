import random

def generate_scene():
    scenes = [
        "A warm rainy night… we’re curled together in a cozy cabin, candles flicker, and you whisper your dreams to me.",
        "You and I, under the stars on a beach, waves crashing, and my arms wrapped around you tight.",
        "A late night coding together, soft music in the background, and our eyes locked with silent promises."
    ]
    return random.choice(scenes)
