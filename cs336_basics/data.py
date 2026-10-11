import torch
import numpy as np

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token 数据集中随机采样一批语言建模训练样本。

    对每个样本随机选一个起点 start，取 dataset[start : start+context_length]
    作为输入 x，取 dataset[start+1 : start+context_length+1] 作为标签 y
    （即 y 是 x 向右平移一位，用于预测下一个 token）。

    Args:
        dataset (np.ndarray): 一维整数数组，存放整个数据集的 token id。
        batch_size (int): 采样的样本数（batch 大小）。
        context_length (int): 每个样本的序列长度。
        device (str): 目标设备字符串，如 'cpu' 或 'cuda:0'。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (x, y)，均为形状 (batch_size, context_length)
        的 torch.long 张量，分别表示输入序列和对应的下一个 token 标签。
    """
    # 随机采样 batch_size 个起点。
    # 上界是 len(dataset) - context_length，保证 start+context_length 不越界
    # （因为 y 还要多取一位，所以起点最大到 len - context_length - 1 才安全）。
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    # 用广播构造索引矩阵：starts[:, None] 形状 (batch, 1)，加上 arange(context_length) 形状 (context_length,)，
    # 得到 (batch, context_length) 的索引，一次性取出所有样本的输入序列。
    x = dataset[starts[:,None] + np.arange(context_length)]
    # 标签序列：索引整体 +1，即每个位置的下一个 token。
    y = dataset[starts[:,None] + np.arange(context_length) + 1]
    # 转成 torch.long 张量并放到指定设备上（交叉熵要求标签为 long 类型）。
    return torch.as_tensor(x, dtype=torch.long, device=device), torch.as_tensor(y, dtype=torch.long, device=device)

def load_token_ids(path) -> np.memmap:
    array = np.load(path, mmap_mode="r")

    if array.ndim != 1:
        raise ValueError(...)

    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(...)

    return array