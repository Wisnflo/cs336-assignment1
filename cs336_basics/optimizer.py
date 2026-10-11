import torch,math
from collections.abc import Iterable

class AdamW(torch.optim.Optimizer):
    """AdamW 优化器 (Adam with decoupled Weight Decay)。

    与标准 Adam 的区别：权重衰减(weight decay)不再混进梯度里，
    而是直接对参数做乘性收缩 p <- p * (1 - lr * weight_decay)，
    这样衰减强度不受自适应学习率的影响（见 Loshchilov & Hutter, 2019）。

    每个参数维护两个状态：
        exp_avg    (m)：梯度的一阶动量（指数滑动平均）
        exp_avg_sq (v)：梯度平方的二阶动量
    更新公式（t 为步数）：
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        m_hat = m / (1 - beta1^t)          # 偏差修正
        v_hat = v / (1 - beta2^t)          # 偏差修正
        p = p * (1 - lr * weight_decay)    # 解耦权重衰减
        p = p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,  
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        """
        Args:
            params (Iterable[torch.nn.Parameter]): 待优化的参数（或参数组）。
            lr (float): 学习率。
            betas (tuple[float, float]): 一阶/二阶动量的衰减系数 (beta1, beta2)。
            eps (float): 加到分母上的小量，保证数值稳定。
            weight_decay (float): 解耦权重衰减系数。
        """
        # 把超参数打包成默认参数组，交给基类 Optimizer 管理。
        default = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, default)

    def step(self) -> None:
        """执行一步参数更新（遍历所有参数组和参数）。"""
        # 遍历每个参数组（不同组可以有不同的 lr/betas 等超参数）。
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                # 没有梯度的参数跳过（例如被冻结的参数）。
                if p.grad is None:
                    continue
                # self.state[p] 是该参数专属的状态字典，用于保存动量等。
                d = self.state[p]
                # 首次遇到该参数时，初始化步数和两个动量（与参数同形状的全零张量）。
                if "step" not in d:
                    d["step"] = 0
                    d["exp_avg"] = torch.zeros_like(p)
                    d["exp_avg_sq"] = torch.zeros_like(p)
                d["step"] += 1
                g = p.grad
                m = d["exp_avg"]
                v = d["exp_avg_sq"]
                # 更新一阶/二阶动量（原地操作，不建计算图）。
                #   m = beta1 * m + (1 - beta1) * g
                #   v = beta2 * v + (1 - beta2) * g^2
                with torch.no_grad():
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).add_(g ** 2, alpha=1 - beta2)
                # 取出本组的超参数。
                lr , eps , weight_decay = group["lr"] , group["eps"] , group["weight_decay"]
                t = d["step"]
                # 偏差修正：早期步数下动量偏小，除以 (1 - beta^t) 校正。
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                with torch.no_grad():
                    # 解耦权重衰减：直接对参数做乘性收缩。
                    p.mul_(1 - lr * weight_decay)
                    # 自适应更新量：lr * m_hat / (sqrt(v_hat) + eps)。
                    adaptive_update = lr * m_hat / (v_hat.sqrt() + eps)
                    # 更新参数：p = p - adaptive_update。
                    p.sub_(adaptive_update)

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """带线性预热的余弦学习率调度 (cosine schedule with linear warmup)。

    分三段：
        1) 预热阶段 (it < warmup_iters)：
               lr = max_lr * it / warmup_iters        # 从 0 线性升到 max_lr
        2) 余弦退火阶段 (warmup_iters <= it < cosine_cycle_iters)：
               lr = min_lr + (max_lr - min_lr) * (1 + cos(pi * r)) / 2
               其中 r = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        3) 退火结束后 (it >= cosine_cycle_iters)：
               lr = min_lr

    Args:
        it (int): 当前迭代步数。
        max_learning_rate (float): 峰值学习率 alpha_max。
        min_learning_rate (float): 最终/最小学习率 alpha_min。
        warmup_iters (int): 线性预热的步数 T_w。
        cosine_cycle_iters (int): 余弦退火的总步数 T_c。

    Returns:
        float: 当前迭代步数对应的学习率。
    """
    # 阶段 1：线性预热，从 0 线性增长到 max_learning_rate。
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # 阶段 2：余弦退火，从 max_learning_rate 平滑降到 min_learning_rate。
    elif it < cosine_cycle_iters:
        # r 是退火进度，从 0 到 1。
        r = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # 直接用 math.cos 对 Python float 计算，保持 float64 精度。
        # 若写成 math.cos(torch.tensor(...))，会先转成 float32 张量，精度丢失导致测试失败。
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + math.cos(r * math.pi)) / 2
    # 阶段 3：退火结束后保持最小学习率。
    else:
        return min_learning_rate