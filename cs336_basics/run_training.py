import argparse
import math
from pathlib import Path

import numpy as np

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import load_token_ids
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.train import train_loop


def non_negative_int(value: str) -> int:
    """argparse 类型转换器：把命令行字符串解析为「非负整数」。

    用作 add_argument(type=...) 时，argparse 会捕获 ArgumentTypeError，
    打印错误信息并以退出码 2 结束，从而拒绝非法输入。

    Args:
        value (str): 命令行传入的原始字符串。

    Returns:
        int: 解析后的非负整数。

    Raises:
        argparse.ArgumentTypeError: 当字符串不是整数，或解析结果小于 0 时。
    """
    try:
        parsed = int(value)
    except ValueError:
        # 不是合法整数（例如 "abc"）。
        raise argparse.ArgumentTypeError(f"不是合法整数：{value!r}")
    if parsed < 0:
        # 是整数但为负数（例如 "-1"）。
        raise argparse.ArgumentTypeError(f"必须是非负整数，收到：{parsed}")
    return parsed


def positive_int(value: str) -> int:
    """argparse 类型转换器：把命令行字符串解析为「正整数」。

    与 non_negative_int 的区别是这里要求严格大于 0，
    用于 batch_size、context_length 等不能为 0 的参数。

    Args:
        value (str): 命令行传入的原始字符串。

    Returns:
        int: 解析后的正整数。

    Raises:
        argparse.ArgumentTypeError: 当字符串不是整数，或解析结果小于等于 0 时。
    """
    try:
        parsed = int(value)
    except ValueError:
        # 不是合法整数（例如 "abc"）。
        raise argparse.ArgumentTypeError(f"不是合法整数：{value!r}")
    if parsed <= 0:
        # 是整数但非正（例如 "0"、"-1"）。
        raise argparse.ArgumentTypeError(f"必须是正整数，收到：{parsed}")
    return parsed


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """定义并解析训练脚本的全部命令行参数。

    参数分为几类：数据路径、训练步数/批大小、模型结构、优化器超参数、
    学习率调度、评估与 checkpoint 保存、日志输出。

    Args:
        argv (list[str] | None): 待解析的参数列表；为 None 时读取 sys.argv。

    Returns:
        argparse.Namespace: 解析后的参数命名空间，属性名即去掉 "--" 后的参数名。
    """
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    # 数据相关：训练集必填，验证集可选。
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, default=None)
    # 训练规模：步数允许为 0（用于只做校验/续训检查），批大小必须为正。
    parser.add_argument("--steps", type=non_negative_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--context-length", type=positive_int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    # 模型结构：词表大小、隐藏维度、层数、注意力头数、前馈维度、RoPE 底数。
    parser.add_argument("--vocab-size", type=positive_int, default=10000)
    parser.add_argument("--d-model", type=positive_int, default=64)
    parser.add_argument("--num-layers", type=positive_int, default=2)
    parser.add_argument("--num-heads", type=positive_int, default=4)
    parser.add_argument("--d-ff", type=positive_int, default=192)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    # 优化器超参数：峰值学习率、AdamW 的 beta1/beta2/eps、解耦权重衰减。
    parser.add_argument("--max-learning-rate", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    # 学习率调度：最小学习率、线性预热步数、余弦退火总步数。
    parser.add_argument("--min-learning-rate", type=float, default=0.0001)
    parser.add_argument("--warmup-iters", type=non_negative_int, default=10)
    parser.add_argument("--cosine-cycle-iters",type=positive_int, default=100)
    # 评估：每隔多少步评估一次、每次评估用多少个 batch。
    parser.add_argument("--eval-every", type=positive_int, default=10)
    parser.add_argument("--num-eval-batches", type=positive_int, default=10)
    # checkpoint：保存目录、保存间隔、从哪个 checkpoint 恢复。
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--save-every", type=positive_int, default=100)
    parser.add_argument("--resume-from",type=Path, default=None)
    # 日志：每隔多少步打印一次训练进度。
    parser.add_argument("--log-every", type=positive_int, default=10)
    return parser.parse_args(argv)

def validate_model_args(args: argparse.Namespace) -> None:
    """校验模型结构相关参数是否自洽。

    检查 d_model 能被 num_heads 整除、每个头的维度 d_head 为偶数
    （RoPE 需要成对旋转），以及 rope_theta 为有限正数。

    Args:
        args (argparse.Namespace): 解析后的命令行参数。

    Raises:
        ValueError: 任一约束不满足时。
    """
    d_model = args.d_model
    num_heads = args.num_heads
    rope_theta = args.rope_theta

    # 每个注意力头的维度；必须能整除，否则多头切分不合法。
    d_head = d_model // num_heads
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    # RoPE 把维度两两配对做旋转，因此 d_head 必须是偶数。
    if d_head % 2 != 0:
        raise ValueError(f"d_head ({d_head}) must be divisible by 2")
    # rope_theta 参与频率计算，必须是有限正数。
    if not math.isfinite(rope_theta):
        raise ValueError(f"rope_theta ({rope_theta}) must be finite")
    if rope_theta <= 0:
        raise ValueError(f"rope_theta ({rope_theta}) must be positive")


def validate_optimizer_args(args: argparse.Namespace) -> None:
    """校验 AdamW 优化器超参数的取值范围。

    要求各超参数均为有限值，且满足：峰值学习率 > 0、beta1/beta2 在 [0, 1)、
    eps > 0、weight_decay >= 0。

    Args:
        args (argparse.Namespace): 解析后的命令行参数。

    Raises:
        ValueError: 任一约束不满足时。
    """
    max_learing_rate = args.max_learning_rate
    beta1 = args.beta1
    beta2 = args.beta2
    eps = args.eps
    weight_decay = args.weight_decay

    # 先统一检查是否为有限值（排除 inf / nan）。
    if not math.isfinite(max_learing_rate):
        raise ValueError(f"max_learning_rate ({max_learing_rate}) must be finite")
    if not math.isfinite(beta1):
        raise ValueError(f"beta1 ({beta1}) must be finite")
    if not math.isfinite(beta2):
        raise ValueError(f"beta2 ({beta2}) must be finite")
    if not math.isfinite(eps):
        raise ValueError(f"eps ({eps}) must be finite")
    if not math.isfinite(weight_decay):
        raise ValueError(f"weight_decay ({weight_decay}) must be finite")
    # 再检查取值范围。
    if max_learing_rate <= 0:
        raise ValueError(f"max_learning_rate ({max_learing_rate}) must be positive")
    # beta 是动量衰减系数，必须落在 [0, 1) 内，否则动量会发散或恒为 0。
    if beta1 < 0 or beta1 >= 1:
        raise ValueError(f"beta1 ({beta1}) must be between 0 and 1")
    if beta2 < 0 or beta2 >= 1:
        raise ValueError(f"beta2 ({beta2}) must be between 0 and 1")
    # eps 作为分母保护项必须为正。
    if eps <= 0:
        raise ValueError(f"eps ({eps}) must be positive")
    # 权重衰减系数不能为负（负值相当于放大权重）。
    if weight_decay < 0:
        raise ValueError(f"weight_decay ({weight_decay}) must be non-negative")


def validate_schedule_args(args: argparse.Namespace) -> None:
    """校验学习率调度与评估相关参数。

    要求 min_learning_rate 有限且落在 [0, max_learning_rate]，
    warmup_iters 落在 [0, cosine_cycle_iters)，评估参数非负。

    Args:
        args (argparse.Namespace): 解析后的命令行参数。

    Raises:
        ValueError: 任一约束不满足时。
    """
    if not math.isfinite(args.min_learning_rate):
        raise ValueError(f"min_learning_rate ({args.min_learning_rate}) must be finite")
    # 最小学习率不能超过峰值学习率，否则调度曲线无意义。
    if args.min_learning_rate < 0 or args.min_learning_rate > args.max_learning_rate:
        raise ValueError(f"min_learning_rate ({args.min_learning_rate}) must be between 0 and max_learning_rate ({args.max_learning_rate})")
    # 预热步数必须小于余弦退火总步数，否则退火阶段为空。
    if args.warmup_iters < 0 or args.warmup_iters >= args.cosine_cycle_iters:
        raise ValueError(f"warmup_iters ({args.warmup_iters}) must be between 0 and cosine_cycle_iters ({args.cosine_cycle_iters})")
    if args.eval_every < 0:
        raise ValueError(f"eval_every ({args.eval_every}) must be non-negative")
    if args.num_eval_batches < 0:
        raise ValueError(f"num_eval_batches ({args.num_eval_batches}) must be non-negative")


def validate_dataset_length(
    dataset: np.ndarray,
    name: str,
    context_length: int,
) -> int:
    """校验单个数据集长度是否足够切出一个训练样本。

    一个样本需要 context_length 个输入 token 加 1 个预测目标 token，
    因此 token 总数至少为 context_length + 1。

    Args:
        dataset (np.ndarray): token id 序列。
        name (str): 数据集名称，用于错误信息（如 "Training"）。
        context_length (int): 上下文长度。

    Returns:
        int: 数据集的 token 总数。

    Raises:
        ValueError: token 数少于 context_length + 1 时。
    """
    tokens = len(dataset)
    if tokens < context_length + 1:
        raise ValueError(f"{name} dataset has {tokens} tokens, which is less than the required context length + 1 ({context_length + 1})")
    return tokens


def validate_args(args: argparse.Namespace) -> None:
    """汇总执行所有参数校验（模型、优化器、调度）。"""
    validate_model_args(args)
    validate_optimizer_args(args)
    validate_schedule_args(args)


def load_datasets(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None]:
    """加载训练集与（可选的）验证集 token 序列。

    Args:
        args (argparse.Namespace): 解析后的命令行参数。

    Returns:
        tuple[np.ndarray, np.ndarray | None]: (训练集, 验证集)；
            未提供 --val-data 时验证集为 None。
    """
    train_dataset = load_token_ids(args.train_data)
    # 验证集可选：只有显式传入路径时才加载。
    val_dataset = load_token_ids(args.val_data) if args.val_data is not None else None
    return train_dataset, val_dataset


def validate_datasets(
    train_dataset: np.ndarray,
    val_dataset: np.ndarray | None,
    context_length: int,
) -> None:
    """校验训练集（以及存在时的验证集）长度是否满足上下文要求。"""
    validate_dataset_length(train_dataset, "Training", context_length)
    if val_dataset is not None:
        validate_dataset_length(val_dataset, "Validation", context_length)


def build_model(args: argparse.Namespace) -> TransformerLM:
    """根据命令行参数构造 TransformerLM 模型。"""
    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )


def build_optimizer(
    model: TransformerLM,
    args: argparse.Namespace,
) -> AdamW:
    """根据命令行参数为模型构造 AdamW 优化器。

    注意这里用 max_learning_rate 作为初始学习率，
    实际训练中每步的学习率由余弦调度覆盖。
    """
    return AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

def build_checkpoint_dir(
    args: argparse.Namespace,
) -> str | None:
    """把 checkpoint 目录参数转成字符串；未指定时返回 None。"""
    if args.checkpoint_dir is None:
        return None
    return str(args.checkpoint_dir)

def run_training(
    model: TransformerLM,
    optimizer: AdamW,
    train_dataset: np.ndarray,
    val_dataset: np.ndarray | None,
    args: argparse.Namespace,
    start_step:int = 0,
) -> tuple[list[float], list[tuple[int, float]]]:
    """调用底层训练循环，把命令行参数展开成 train_loop 的入参。

    Args:
        model (TransformerLM): 待训练模型。
        optimizer (AdamW): 优化器。
        train_dataset (np.ndarray): 训练集 token 序列。
        val_dataset (np.ndarray | None): 验证集 token 序列，可为 None。
        args (argparse.Namespace): 解析后的命令行参数。
        start_step (int): 起始步数（续训时不为 0）。

    Returns:
        tuple[list[float], list[tuple[int, float]]]:
            (每步训练损失, [(步数, 验证损失), ...])。
    """
    return train_loop(
        model=model,
        optimizer=optimizer,
        dataset=train_dataset,
        val_dataset=val_dataset,
        steps=args.steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        eval_every=args.eval_every,
        num_eval_batches=args.num_eval_batches,
        checkpoint_dir=build_checkpoint_dir(args),
        save_every=args.save_every,
        start_step=start_step,
        log_every=args.log_every,
    )

def restore_training_state(
    model: TransformerLM,
    optimizer: AdamW,
    args: argparse.Namespace,
) -> int:
    """从 checkpoint 恢复模型与优化器状态，返回已完成的步数。

    未指定 --resume-from 时视为从头训练，返回 0。

    Args:
        model (TransformerLM): 模型（会被就地加载参数）。
        optimizer (AdamW): 优化器（会被就地加载状态）。
        args (argparse.Namespace): 解析后的命令行参数。

    Returns:
        int: checkpoint 中记录的已完成步数。

    Raises:
        ValueError: checkpoint 中的步数非法（为负或超过目标步数）时。
    """
    if args.resume_from is None:
        return 0
    iteration = load_checkpoint(
        args.resume_from,
        model,
        optimizer,
    )
    # 步数不能为负，也不能超过本次训练的目标步数。
    if iteration < 0 or iteration > args.steps:
        raise ValueError(f"Invalid iteration count from checkpoint: {iteration}")
    return iteration

def save_final_checkpoint(
    model: TransformerLM,
    optimizer: AdamW,
    iteration: int,
    checkpoint_dir: Path,
) -> Path:
    """把最终状态保存为 checkpoint_dir/final.pt，并返回该路径。"""
    # 目录可能不存在，先递归创建。
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "final.pt"
    save_checkpoint(
        model,
        optimizer,
        iteration,
        checkpoint_path,
    )
    return checkpoint_path

def save_final_state(
    model: TransformerLM,
    optimizer: AdamW,
    losses: list[float],
    start_step: int,
    args: argparse.Namespace,
) -> Path | None:
    """训练结束后保存最终 checkpoint；无需保存时返回 None。

    没有产生任何训练步（losses 为空）或未指定 checkpoint 目录时跳过保存。
    """
    if not losses or args.checkpoint_dir is None:
        return None
    # 最终步数 = 起始步数 + 本次实际训练步数。
    return save_final_checkpoint(
        model,
        optimizer,
        start_step + len(losses),
        args.checkpoint_dir,
    )

def report(
    model: TransformerLM,
    optimizer: AdamW,
    losses: list[float],
    val_losses: list[tuple[int, float]],
    start_step:int,
    final_checkpoint: Path | None,
) -> None:
    """以 key=value 形式打印训练摘要，供测试脚本解析。

    输出字段包括参数量、优化器、完成步数、最终训练损失、当前学习率、
    各次验证损失、起始步数、本次新增步数以及最终 checkpoint 路径。
    """
    # 统计模型总参数量。
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"model_params={num_params}")
    print(f"optimizer={optimizer}")
    print(f"completed_steps={start_step + len(losses)}")
    # losses 为空时（零步训练）最终损失记为 None。
    print(f"final_train_loss={losses[-1] if losses else None}")
    print(f"current_learning_rate={optimizer.param_groups[0]['lr']}")
    for step, val_loss in val_losses:
        print(f"validation_step={step} validation_loss={val_loss}")
    print(f"start_step={start_step}")
    print(f"new_steps={len(losses)}")
    print(f"final_checkpoint={final_checkpoint}")


def main() -> None:
    """训练脚本入口：解析参数 → 校验 → 加载数据 → 建模 → 训练 → 保存 → 汇报。"""
    args = parse_args()

    # 1) 先做参数校验，尽早暴露非法配置。
    validate_args(args)

    # 2) 加载并校验数据集长度。
    train_dataset, val_dataset = load_datasets(args)
    validate_datasets(train_dataset, val_dataset, args.context_length)

    # 3) 构建模型与优化器，并按需从 checkpoint 恢复状态。
    model = build_model(args)
    optimizer = build_optimizer(model, args)
    start_step = restore_training_state(model, optimizer, args)

    # 4) 执行训练循环。
    losses, val_losses = run_training(
        model, optimizer, train_dataset, val_dataset, args, start_step
    )
    # 5) 保存最终 checkpoint 并打印摘要。
    final_checkpoint = save_final_state(model, optimizer, losses, start_step, args)

    report(model, optimizer, losses, val_losses, start_step, final_checkpoint)


if __name__ == "__main__":
    main()