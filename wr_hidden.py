import torch

from torch import Tensor


def cosine_similarity(first: Tensor, second: Tensor) -> Tensor:
    first_norm = first / torch.linalg.norm(first, dim=1, keepdim=True)
    second_norm = second / torch.linalg.norm(second, dim=1, keepdim=True)
    return first_norm @ second_norm.T


def margin_based_scoring(
    first: Tensor, second: Tensor, k=4, variant="ratio", eps=1e-6
) -> Tensor:
    a = cosine_similarity(first, second)
    if variant == "absolute":
        return a
    nn_k_row = torch.topk(a, k=min(k, a.size(1)), dim=1).values
    nn_k_col = torch.topk(a, k=min(k, a.size(0)), dim=0).values
    row_mean = nn_k_row.mean(dim=1, keepdim=True)
    col_mean = nn_k_col.mean(dim=0, keepdim=True)
    b = (row_mean + col_mean) * 0.5 + eps
    return a - b if variant == "distance" else a / b


def layer_accuracy(score: Tensor) -> float:
    device = score.device
    preds = score.argmax(dim=1)
    return (preds == torch.arange(score.size(0), device=device)).float().mean().item()


def mean_hidden_state(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    count = mask.sum(dim=1)
    return summed / count


def last_token_hidden_state(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Take the hidden state of the last non-padding token for each sequence.
    hidden_states (b, l, h)
    attention_mask (b, l)
    => (b, h)
    """
    # (b)
    last_token_index = attention_mask.ne(0).sum(dim=1) - 1
    # (b)
    batch_range = torch.arange(hidden_states.size(0), device=hidden_states.device)
    # (b, h)
    return hidden_states[batch_range, last_token_index]


@torch.no_grad()
def layer_representation(
    model,
    tokenizer,
    sentences: list[str],
    batch_size=32,
    device="cuda",
    hidden_state_fn=mean_hidden_state,
) -> dict[int, Tensor]:
    temp: dict[int, list[Tensor]] = {}
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        hidden_states = model(
            **tokens, output_hidden_states=True, use_cache=False
        ).hidden_states
        for i, hs in enumerate(hidden_states):
            hidden_state = hidden_state_fn(hs, tokens["attention_mask"])
            temp.setdefault(i, []).append(hidden_state)
        del hidden_states, tokens
        torch.cuda.empty_cache()
    return {i: torch.cat(temp[i], dim=0) for i in temp}
