import torch


def build_story_ids(token_ids, eos_id):
    eos_flags = (token_ids == eos_id).to(torch.int32)

    end_counts = torch.cumsum(
        eos_flags,
        dim=1,
        dtype=torch.int32,
    )

    return end_counts - eos_flags

def build_same_story_mask(story_ids):
    query_story_ids = story_ids.unsqueeze(2)
    key_story_ids = story_ids.unsqueeze(1)

    return query_story_ids == key_story_ids
