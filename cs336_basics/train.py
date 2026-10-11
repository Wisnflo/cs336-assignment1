from pathlib import Path
from time import perf_counter

from cs336_basics.model import TransformerLM, cross_entropy,clip_gradients
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW,get_lr_cosine_schedule
from cs336_basics.checkpoint import save_checkpoint,load_checkpoint
import torch
import numpy as np

def language_model_loss(
    model: TransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logits = model(x)
    logits = logits.reshape(-1, logits.size(-1))
    y = y.reshape(-1)
    return cross_entropy(logits, y)
    
def train_step(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    max_grad_norm: float = 1.0,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = language_model_loss(model, x, y)
    loss.backward()
    clip_gradients(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item()

def log_training_progress(
    step: int,
    loss: float,
    learning_rate: float,
    elapsed_seconds: float,
) -> None:
    print(f"step={step} train_loss={loss:.6f} learning_rate={learning_rate:.6g} elapsed_seconds={elapsed_seconds:.3f}",flush=True)

def train_loop(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    dataset:np.ndarray,
    steps: int,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    val_dataset: np.ndarray | None = None,
    eval_every: int | None = None,
    num_eval_batches: int | None = None,
    checkpoint_dir: str | None = None,
    save_every: int | None = None,
    start_step: int = 0,
    log_every:int | None = None,
)-> tuple[list[float], list[tuple[int, float]]]:
    if start_step < 0 or start_step > steps:
        raise ValueError("start_step must be a non-negative integer")
    if save_every is not None and save_every <= 0:
        raise ValueError("save_every must be a positive integer")
    if log_every is not None and log_every <= 0:
        raise ValueError("log_every must be a positive integer")

    losses = []
    val_losses = []
    started_at = perf_counter()
    for step in range(start_step, steps):
        model.train()
        lr = get_lr_cosine_schedule(step,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x,y = get_batch(dataset, batch_size, context_length, device)
        loss = train_step(model, optimizer, x, y)
        losses.append(loss)

        if val_dataset is not None and eval_every is not None and (step+1) % eval_every == 0:
            if num_eval_batches is None:
                num_eval_batches = 10
            val_loss = evaluate_loss(model, val_dataset, batch_size, context_length, device, num_eval_batches)
            val_losses.append((step+1, val_loss))

        if checkpoint_dir is not None and save_every is not None and (step+1) % save_every == 0:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            file_path = checkpoint_path / f"step-{step+1}.pt"
            save_checkpoint(model, optimizer, step+1, file_path)

        if log_every is not None and ((step+1) % log_every == 0 or (step+1) == steps):
            elapsed_seconds = perf_counter() - started_at
            log_training_progress(step+1, loss, lr, elapsed_seconds)

    return losses, val_losses
        
def evaluate_loss(
    model:TransformerLM,
    dataset:np.ndarray,
    batch_size:int,
    context_length:int,
    device:torch.device | str,
    num_batches:int = 10,
) -> float:
    training_pattern = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x,y = get_batch(dataset, batch_size, context_length, device)
            loss = language_model_loss(model, x, y)
            losses.append(loss.item())  
    model.train(training_pattern)
    return float(np.mean(losses))

