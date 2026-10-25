import torch


def generate(model, tokenizer, prompt, config, device, seed):
    settings = config["generation"]

    max_context = config["model"]["max_context"]
    max_new_tokens = settings["max_new_tokens"]
    temperature = settings["temperature"]
    top_k = settings["top_k"]

    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    if not 1 <= top_k <= tokenizer.get_vocab_size():
        raise ValueError("top_k must be within the vocabulary size.")

    eos_id = tokenizer.token_to_id(
        config["tokenizer"]["eos_token"]
    )

    prompt_ids = tokenizer.encode(prompt).ids

    if not prompt_ids:
        prompt_ids = [eos_id]

    ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    previous_training_mode = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = ids[:, -max_context:]

                logits, _ = model(context)

                scores = logits[:, -1, :].float()
                scores = scores / temperature

                top_scores = torch.topk(
                    scores,
                    k=top_k,
                    dim=-1,
                ).values

                threshold = top_scores[:, -1:]

                scores = scores.masked_fill(
                    scores < threshold,
                    float("-inf"),
                )

                probabilities = torch.softmax(
                    scores,
                    dim=-1,
                )

                next_id = torch.multinomial(
                    probabilities.cpu(),
                    num_samples=1,
                    generator=rng,
                ).to(device)

                ids = torch.cat(
                    [ids, next_id],
                    dim=1,
                )

                if next_id.item() == eos_id:
                    break

    finally:
        model.train(previous_training_mode)

    return tokenizer.decode(
        ids[0].tolist(),
        skip_special_tokens=True,
    )