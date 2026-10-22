from pathlib import Path

from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
)


def train_tokenizer(train_texts, config):
    settings = config["tokenizer"]

    tokenizer = Tokenizer(
        models.BPE(unk_token=settings["unk_token"])
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    bpe_trainer = trainers.BpeTrainer(
        vocab_size=settings["vocab_size"],
        min_frequency=settings["min_frequency"],
        special_tokens=[
            settings["unk_token"],
            settings["eos_token"],
        ],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(
        train_texts,
        trainer=bpe_trainer,
    )

    path = Path(settings["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))

    return tokenizer