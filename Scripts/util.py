import random


def color_for_id(global_id: int):
    random.seed(global_id)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )
