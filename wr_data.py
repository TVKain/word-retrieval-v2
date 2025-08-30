import random
from datasets import load_dataset

def sample_pairs(
    base: list[str], target: list[str], sample_size: int
) -> tuple[list[str], list[str]]:
    """
    Randomly sample word pairs from full lists while keeping alignment.

    Args:
        base: List of base words
        arget: Corresponding list of target words
        sample_size: Number of word pairs to sample

    Returns:
        Tuple of (sampled_base, sampled_target)
    """
    assert len(base) == len(target), "Lists must have the same length"
    sample_size = min(sample_size, len(base))
    indices = random.sample(range(len(base)), sample_size)
    sampled_base = [base[i] for i in indices]
    sampled_target = [target[i] for i in indices]
    return sampled_base, sampled_target


def filter_pairs(base: list[str], target: list[str], tokenizer):
    """
    Filter out pairs that have more than 1 token
    """
    filtered_base = []
    filtered_target = []

    for i, base_word in enumerate(base):
        base_tokens = tokenizer(base_word, return_tensors="pt", padding=False)
        target_tokens = tokenizer(target[i], return_tensors="pt", padding=False)

        if (
            base_tokens["input_ids"].shape[1] == 1
            and target_tokens["input_ids"].shape[1] == 1
        ):
            filtered_base.append(base_word)
            filtered_target.append(target[i])

    return filtered_base, filtered_target

def load_sla_word(lang: str, limit: int = None) -> tuple[list[str], list[str]]:
    """
    Load words from the SLA dataset for a specific language and their base equivalents.

    Args:
        lang (str): ISO language code (e.g., "gle" for Irish, "eus" for Basque, "cmn" for Mandarin Chinese).

    Returns:
        Tuple[List[str], List[str]]:
            - List of words in the target language (specified by `lang`).
            - List of corresponding base words.

    Example:
        target, base = load_sla_word("gle")
        print(target[:5])
        ['agus', 'ar', 'bain', 'bean', 'beidh']
        print(base[:5])
        ['and', 'on', 'use', 'woman', 'will be']
    """
    SLA_WORD = "tvkain/sla"  # Hugging Face dataset repo path
    # Load the dataset (train split)
    ds = load_dataset(SLA_WORD, split="train")

    # Filter the dataset for the given language
    filtered_ds = ds.filter(
        lambda x: x["lang"]
        == lang
    )

    # Extract the target language words and their base equivalents
    target = [entry["word"] for entry in filtered_ds]
    base = [entry["eng"] for entry in filtered_ds]

    if limit:
        target = target[:limit]
        base = base[:limit]

    return base, target
