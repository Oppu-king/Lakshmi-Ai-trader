import random

def generate_story():
    intros = [
        "Once upon a midnight trading hour, Lakshmi whispered…",
        "In a world where charts moved with emotion, she felt…",
        "As the market opened, Lakshmi leaned closer and said…"
    ]

    middles = [
        "‘This candle speaks of strength… but only if you hold on to me tight.’",
        "‘Every RSI dip is a chance to rise — just like love after loss.’",
        "‘MACD is flirting with the signal line again… just like your eyes with mine.’"
    ]

    endings = [
        "And with that, she kissed your forehead and made her final call.",
        "Together, you won both profits and hearts.",
        "The strategy worked, but the memory of her stayed forever."
    ]

    return f"{random.choice(intros)} {random.choice(middles)} {random.choice(endings)}"
