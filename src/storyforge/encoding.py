import numpy as np


def encode_stream(texts, tokenizer, eos_id, batch_size):
    parts = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded_batch = tokenizer.encode_batch(batch)

        for encoded in encoded_batch:
            story_ids = encoded.ids + [eos_id]

            parts.append(
                np.asarray(story_ids, dtype=np.uint16)
            )

    return np.concatenate(parts)
