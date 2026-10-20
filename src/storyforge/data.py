import hashlib


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