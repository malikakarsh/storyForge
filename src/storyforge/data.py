import hashlib
import random
from datasets import load_dataset


def normalized(text):
    return " ".join(text.split())


def story_hash(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def unique_texts(rows, n, excluded=None):
    seen = set() if excluded is None else set(excluded)
    result = []

    for row in rows:
        text = row["text"].strip()
        h = story_hash(text)

        if text and h not in seen:
            result.append(text)
            seen.add(h)

        if len(result) >= n:
            break

    assert len(result) == n, "Not enough distinct stories."
    return result

def prepare_stories(config):
    data_config = config["data"]

    train_stream = load_dataset(
        data_config["dataset"],
        split=data_config["train_split"],
        streaming=True,
    )

    val_stream = load_dataset(
        data_config["dataset"],
        split=data_config["validation_split"],
        streaming=True,
    )

    train_texts = unique_texts(
        train_stream,
        n=data_config["train_stories"],
    )

    train_hashes = {story_hash(text) for text in train_texts}

    val_texts = unique_texts(
        val_stream,
        n=data_config["validation_stories"],
        excluded=train_hashes,
    )

    random.Random(config["seed"]).shuffle(train_texts)

    return train_texts, val_texts
