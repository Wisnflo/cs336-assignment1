import torch,math

from cs336_basics.model import softmax,TransformerLM
from cs336_basics.tokenizer import Tokenizer

def filter_top_p(
    probs: torch.Tensor,
    top_p: float = 1.0,
) -> torch.Tensor:
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D tensor")
    if probs.shape[0] <= 0 or probs.shape[1] <= 0:
        raise ValueError("probs must have at least one row and one column")
    if not math.isfinite(top_p):
        raise ValueError("top_p must be a finite number")
    if top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be a number between 0 and 1")
    if top_p == 1.0:
        return probs.clone()
    
    sorted_probs, sorted_indices = torch.sort(
        probs,
        dim=-1,
        descending=True,
    )
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    prev = torch.zeros_like(cumulative)
    prev[:,1:] = cumulative[:,:-1]
    mask = prev >= top_p
    normalized_sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    normalized_sorted_probs = normalized_sorted_probs / normalized_sorted_probs.sum(dim=-1,keepdim=True)
    restored = torch.zeros_like(probs).scatter(
        dim=-1,
        index=sorted_indices,
        src=normalized_sorted_probs,
    )
    return restored

def generate_tokens(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None=None,
) -> torch.Tensor:
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must be a 2D tensor")
    if prompt_ids.shape[0] != 1 or prompt_ids.shape[1] <= 0:
        raise ValueError("prompt_ids must have exactly one row and at least one column")
    if prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must be a tensor of type torch.long")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if eos_token_id is not None and not isinstance(eos_token_id, int):
        raise ValueError("eos_token_id must be an integer or None")
    if eos_token_id is not None and eos_token_id < 0:
        raise ValueError("eos_token_id must be a non-negative integer")

    generated = prompt_ids.clone()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = generated[:, -context_length:]
                logits = model(context)
                next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id == next_token.item():
                    break
    finally:
        model.train(mode=was_training)

    return generated


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor")
    if logits.shape[0] <= 0 or logits.shape[1] <= 0:
        raise ValueError("logits must have at least one row and one column")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be a finite number")
    if temperature <= 0:
        raise ValueError("temperature must be a positive number")

    weights = softmax(logits / temperature,dim = -1)
    weights = filter_top_p(weights, top_p=top_p)
    return torch.multinomial(weights, num_samples=1)

def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str | None = "<|endoftext|>",
) -> str:
    encoded_ids = tokenizer.encode(prompt)
    if not encoded_ids:
        raise ValueError("Failed to encode the prompt")

    eos_token_id = None
    if eos_token is not None and not eos_token in tokenizer.special_tokens:
        raise ValueError("eos_token is not a special token in the tokenizer")

    if eos_token is not None and eos_token in tokenizer.special_tokens:
        eos_token_id = tokenizer.token_to_id[eos_token.encode("utf-8")]

    device = next(model.parameters()).device
    input = torch.tensor([encoded_ids],dtype=torch.long,device=device)

    output = generate_tokens(
        model=model,
        prompt_ids=input,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )[0]

    return tokenizer.decode(output.cpu().tolist())

