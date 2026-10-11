import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
) -> None:
    """保存训练检查点 (checkpoint) 到文件。

    把模型参数、优化器状态和当前迭代步数打包成一个字典保存，
    这样之后可以从任意一步完整恢复训练（包括 AdamW 的动量等状态）。

    Args:
        model (torch.nn.Module): 要保存的模型。
        optimizer (torch.optim.Optimizer): 要保存的优化器（含动量等状态）。
        iteration (int): 当前训练迭代步数。
        out: 保存目标，可以是文件路径(str)或可写的类文件对象。

    Returns:
        None
    """
    # state_dict() 返回 "参数名 -> 张量" 的字典，是 PyTorch 标准的序列化格式。
    # 同时保存 optimizer.state_dict() 才能恢复动量/步数等优化器内部状态。
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, out)

def load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """从检查点文件恢复模型和优化器状态。

    Args:
        src: 检查点来源，可以是文件路径(str)或可读的类文件对象。
        model (torch.nn.Module): 要把参数加载进去的模型（原地修改）。
        optimizer (torch.optim.Optimizer): 要恢复状态的优化器（原地修改）。

    Returns:
        int: 检查点中保存的迭代步数，用于从该步继续训练。
    """
    # map_location='cpu'：无论检查点原本存在哪种设备上，都先加载到 CPU，
    #   避免在没有 GPU 的机器上加载 GPU 检查点时报错。
    # weights_only=True：只反序列化张量等安全对象，防止加载不可信文件时执行任意代码。
    checkpoint = torch.load(src,map_location=torch.device('cpu'),weights_only=True)
    # 恢复模型参数。
    model.load_state_dict(checkpoint["model"])
    # 恢复优化器状态（动量、步数等）。
    optimizer.load_state_dict(checkpoint["optimizer"])
    # 返回保存时的迭代步数，供训练循环续跑。
    return checkpoint["iteration"]