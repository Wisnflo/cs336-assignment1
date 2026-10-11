import argparse,math,torch,sys
from pathlib import Path

from cs336_basics.run_training import non_negative_int, positive_int, validate_model_args,build_model
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.generation import generate_text

def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained language model.")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--vocab-file",type=Path,required=True)
    parser.add_argument("--merges-file",type=Path,required=True)
    parser.add_argument("--prompt",type=str,required=True)
    parser.add_argument("--vocab-size",type=positive_int,required=True)
    parser.add_argument("--context-length",type=positive_int,required=True)
    parser.add_argument("--d-model",type=positive_int,required=True)
    parser.add_argument("--num-layers",type=positive_int,required=True)
    parser.add_argument("--num-heads",type=positive_int,required=True)
    parser.add_argument("--d-ff",type=positive_int,required=True)
    parser.add_argument("--rope-theta",type=float,default=10000.0)
    parser.add_argument("--max-new-tokens",type=non_negative_int,default=256)
    parser.add_argument("--temperature",type=float,default=1.0)
    parser.add_argument("--top-p",type=float,default=1.0)
    parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--eos-token",type=str,default="<|endoftext|>")
    parser.add_argument("--no-eos",action="store_true",default=False)


    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    validate_model_args(args)

    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError(f"temperature ({args.temperature}) must be a finite positive number")
    if not math.isfinite(args.top_p) or args.top_p <= 0 or args.top_p > 1:
        raise ValueError(f"top_p ({args.top_p}) must be a finite number in (0, 1]")
    if args.prompt == "":
        raise ValueError("prompt must not be an empty string")
    if args.eos_token == "":
        raise ValueError("eos_token must not be an empty string")

def load_tokenizer(args: argparse.Namespace) -> Tokenizer:
    loaded = Tokenizer.from_files(
        str(args.vocab_file),
        str(args.merges_file),
        special_tokens = None,
    )

    if set(loaded.vocab) != set(range(args.vocab_size)):
        raise ValueError("Loaded tokenizer does not have the expected vocabulary size")
    
    for i in range(256):
        if bytes([i]) not in loaded.token_to_id:
            raise ValueError(f"Loaded tokenizer is missing token {i}")

    if args.eos_token.encode("utf-8") not in loaded.token_to_id:
        raise ValueError("Loaded tokenizer is missing the end-of-sequence token")

    return Tokenizer(loaded.vocab,loaded.merges,[args.eos_token])

def load_model(
    args: argparse.Namespace,
) -> tuple[TransformerLM, int]:
    checkpoint = torch.load(args.checkpoint,map_location="cpu", weights_only=True)

    if not isinstance(checkpoint,dict):
        raise ValueError("Loaded model state is not a dictionary")
    if "model" not in checkpoint or "iteration" not in checkpoint:
        raise ValueError("Loaded model state is missing required keys")
    if not isinstance(checkpoint["iteration"], int) or isinstance(checkpoint["iteration"], bool):
        raise ValueError("Loaded model iteration must be an integer")
    if checkpoint["iteration"] < 0:
        raise ValueError("Loaded model iteration is negative")

    model = build_model(args)
    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()
    return model,checkpoint["iteration"]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    tokenizer = load_tokenizer(args)
    model, iteration = load_model(args)

    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token = args.eos_token if not args.no_eos else None,
    )

    print(generated_text)
    print(f"checkpoint_iteration={iteration}", file=sys.stderr)

if __name__ == "__main__":
    main()