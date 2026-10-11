from collections.abc import Iterable

import torch
import math

class Linear(torch.nn.Module):
    """线性层 (Linear / 全连接层)。

    实现 y = x W^T，其中权重 W 的形状为 (out_features, in_features)。
    注意这里权重存成 (out_features, in_features)，所以 forward 里要转置 W^T，
    这样输出维度才是 out_features。

    权重初始化：截断正态分布 (truncated normal)，
        sigma = sqrt(2 / (in_features + out_features))
        取值范围 [-3*sigma, 3*sigma]
    这是 Transformer 论文中常用的初始化方式，能保持前向/反向的方差稳定。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device : torch.device | str | None = None,
        dtype : torch.dtype | None = None
    ) -> None:
        """
        Args:
            in_features (int): 输入特征维度 d_in。
            out_features (int): 输出特征维度 d_out。
            device (torch.device | str | None): 参数所在设备，None 表示使用默认设备。
            dtype (torch.dtype | None): 参数数据类型，None 表示使用默认类型。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        # 用 torch.nn.Parameter 把 weight 注册为可训练参数，
        # 这样它才会出现在 model.parameters() 里并被优化器更新。
        # 形状 (out_features, in_features)，先分配空张量，稍后再初始化。
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # 截断正态分布的标准差，a/b 是截断边界（±3 sigma）。
        sigma = math.sqrt(2 / (in_features + out_features))
        # 用截断正态分布初始化权重张量（原地操作）。
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma,a=-3*sigma, b=3*sigma)

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        """前向传播：y = x @ W^T。

        Args:
            x (torch.Tensor): 输入，形状 (..., in_features)。

        Returns:
            torch.Tensor: 输出，形状 (..., out_features)。
        """
        # 矩阵乘法：x (..., in_features) @ W^T (in_features, out_features) -> (..., out_features)。
        return x @ self.weight.T


class Embedding(torch.nn.Module):
    """词嵌入层 (Embedding)。

    本质是一张查找表：weight 形状 (num_embeddings, embedding_dim)，
    第 i 行就是 token id = i 对应的 embedding 向量。
    权重用截断正态分布 N(0, 1) 初始化，截断范围 [-3, 3]。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Args:
            num_embeddings (int): 词表大小（embedding 表的行数）。
            embedding_dim (int): 每个 embedding 向量的维度。
            device (torch.device | str | None): 参数所在设备。
            dtype (torch.dtype | None): 参数数据类型。
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        # 可训练的 embedding 表，形状 (num_embeddings, embedding_dim)。
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        # 截断正态分布初始化：均值 0、标准差 1、截断到 [-3, 3]。
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1,a=-3,b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """前向传播：按 token id 查表取 embedding 向量。

        Args:
            token_ids (torch.Tensor): token id 张量，形状 (...)，dtype 为整型。

        Returns:
            torch.Tensor: 对应的 embedding，形状 (..., embedding_dim)。
        """
        # Embedding 本质就是一张查找表(lookup table)：
        # self.weight 的形状是 (num_embeddings, embedding_dim)，
        # 第 i 行就是 token id = i 对应的 embedding 向量。
        #
        # 用 token_ids 作为下标去索引 weight，等价于"按 id 取行"：
        #   - token_ids 形状为 (...) 时，返回形状为 (..., embedding_dim)
        #   - 例如 token_ids = [3, 1]，返回 weight[3] 和 weight[1] 两个向量
        # 这里没有矩阵乘法，只是高级索引(gather)，所以非常高效。
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    """均方根归一化 (Root Mean Square Layer Normalization)。

    与 LayerNorm 不同，RMSNorm 不减均值、也不加偏置，只按均方根缩放：
        y = x / sqrt(mean(x^2) + eps) * weight
    其中 weight 是可学习的逐维缩放参数（形状 d_model）。
    计算时统一升到 float32，避免半精度下平方/求和溢出或精度不足。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Args:
            d_model (int): 归一化的特征维度（最后一维大小）。
            eps (float): 加到分母上的小量，保证数值稳定，防止除零。
            device (torch.device | str | None): 参数所在设备。
            dtype (torch.dtype | None): 参数数据类型。
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        # 可学习的缩放参数，初始化为全 1（即初始时不做缩放）。
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：对最后一维做 RMSNorm。

        Args:
            x (torch.Tensor): 输入，形状 (..., d_model)。

        Returns:
            torch.Tensor: 归一化后的输出，形状与输入相同。
        """
        # self.dtype 默认是 None，它也不代表本次输入的 dtype
        in_dtype = x.dtype
        # 在 float32 下做归一化，避免 float16 下平方/求和溢出或精度不足。
        x = x.to(torch.float32)
        # RMS = sqrt(mean(x^2) + eps)，沿最后一维(d_model)计算，keepdim 保持形状便于广播。
        x_norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化后再乘可学习的缩放参数 weight。
        x_normalized = x / x_norm
        res = x_normalized * self.weight
        # 转回原始 dtype，保证输出类型与输入一致。
        return res.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish 激活函数：silu(x) = x * sigmoid(x)。

    Args:
        x (torch.Tensor): 任意形状的输入张量。

    Returns:
        torch.Tensor: 逐元素应用 SiLU 后的结果，形状与输入相同。
    """
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
    """SwiGLU 前馈网络 (Feed-Forward Network)。

    结构（见 Transformer 论文 3.3 节 / GLU Variants 论文）：
        FFN(x) = W2( silu(W1 x) * W3 x )
    其中 W1、W3 把 d_model 升维到 d_ff，W2 再降回 d_model。
    相比普通 FFN 的 ReLU，SwiGLU 用门控机制（silu(W1 x) 作为门）提升表达能力。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Args:
            d_model (int): 输入/输出维度。
            d_ff (int): 内部升维后的维度（通常约为 8/3 * d_model）。
            device (torch.device | str | None): 参数所在设备。
            dtype (torch.dtype | None): 参数数据类型。
        """
        super().__init__()
        self.d_model = d_model  
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        # w1: 门控分支的升维投影 (d_model -> d_ff)
        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        # w2: 降维投影，把 d_ff 映射回 d_model
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
        # w3: 值分支的升维投影 (d_model -> d_ff)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：W2( silu(W1 x) * W3 x )。

        Args:
            x (torch.Tensor): 输入，形状 (..., d_model)。

        Returns:
            torch.Tensor: 输出，形状 (..., d_model)。
        """
        # silu(w1(x)) 是门控信号，与 w3(x) 逐元素相乘后再经 w2 降维。
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(torch.nn.Module):
    """旋转位置编码 (Rotary Positional Embedding, RoPE)。

    把 d_k 维向量两两配对成 d_k/2 组，对第 i 组按位置 m 旋转角度 m * omega_i，
    其中 omega_i = theta ** (-2i / d_k)。旋转只改变向量的方向、不改变模长，
    因此注意力分数天然只依赖相对位置差 (m - n)。

    实现上预缓存 cos/sin 表，forward 时按 token 位置查表，避免重复计算。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Args:
            theta (float): RoPE 的基频参数 Θ（如 10000.0）。
            d_k (int): 要旋转的向量维度（通常是每个注意力头的维度 d_head）。
            max_seq_len (int): 预缓存的最大序列长度。
            device (torch.device | str | None): 缓存张量所在设备。
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # RoPE 把 d_k 维向量两两配对成 d_k/2 组，第 i 组(i=0..d_k/2-1)的角频率为
        #   omega_i = theta ** (-2i / d_k)
        # 这里一次性算出所有 i 的频率向量，形状 (d_k/2,)。
        # 注意：j 应该是"整个向量" [0,2,4,...]，而不是循环里的单个标量。
        j = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        omega = theta ** (-j / d_k)  # (d_k/2,)

        # 位置 m = 0..max_seq_len-1，与频率做外积得到角度矩阵，形状 (max_seq_len, d_k/2)。
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * omega[None, :]  # (max_seq_len, d_k/2)

        # 预缓存 cos/sin，避免每次 forward 重算。
        # persistent=False：这些是常量，不需要存进 state_dict（否则 load_state_dict 会报缺 key）。
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:
        """对输入张量按 token 位置施加旋转。

        Args:
            x (torch.Tensor): 要旋转的 query 或 key，形状 (..., seq_len, d_k)。
            token_positions (torch.Tensor): 每个 token 的绝对位置，形状 (..., seq_len)。

        Returns:
            torch.Tensor: 旋转后的张量，形状与 x 相同 (..., seq_len, d_k)。
        """
        # 输入：
        #   x              形状 (..., seq_len, d_k)，要施加旋转的 query 或 key
        #   token_positions 形状 (..., seq_len)，每个 token 的绝对位置(不一定从 0 开始)
        # 输出：与 x 同形状 (..., seq_len, d_k)

        # 用 token_positions 索引缓存，例如 self.cos_cache[token_positions]。若位置形状为 (..., seq_len)，取出的余弦、正弦形状就是 (..., seq_len, d_k/2)；若位置只有 (seq_len,)，它们也能向批次维广播。
        # 关键：用位置去"查表"，而不是假设位置就是 0..seq_len-1，这样 KV cache 等场景也正确。
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        # 将 x 最后一维拆成相邻数对：x[..., 0::2] 取每对的第一个数，x[..., 1::2] 取第二个数。0::2 的意思是“从 0 开始，每隔 2 个取一个”。
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # 对每对数 (a, b) 按对应位置的角度旋转，分别算出 a·cos − b·sin 和 a·sin + b·cos。
        # 这就是二维旋转矩阵 [[cos,-sin],[sin,cos]] 作用在 (a,b) 上的结果。
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # 按“新 a、新 b、新 a、新 b”的顺序交错拼回去。可以先用 torch.stack((new_a, new_b), dim=-1) 增加一维，再用 .flatten(start_dim=-2) 合并最后两维。不要直接 cat 两组，否则会变成“所有 a、所有 b”，破坏原有配对顺序。
        out = torch.stack((out1,out2),dim=-1).flatten(start_dim=-2)
        # 转回输入 dtype（cache 是 float32，输入可能是 float16/bfloat16）。
        return out.to(x.dtype)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """数值稳定的 softmax。

    先减去该维最大值再取指数，避免 exp 溢出：
        softmax(x)_i = exp(x_i - max) / sum_j exp(x_j - max)

    Args:
        x (torch.Tensor): 输入张量。
        dim (int): 沿哪个维度做归一化。

    Returns:
        torch.Tensor: 归一化后的张量，形状与输入相同，沿 dim 求和为 1。
    """
    # 减去最大值（keepdim 保持维度以便广播），保证数值稳定。
    max_values = torch.amax(x, dim=dim, keepdim=True)
    exp = torch.exp(x - max_values)
    # 沿 dim 求和作为分母，得到归一化权重。
    sum = torch.sum(exp,dim=dim,keepdim=True)
    return exp/sum

def scaled_dot_product_attention(
    Q : torch.Tensor,
    K : torch.Tensor,
    V : torch.Tensor,
    mask : torch.Tensor | None = None
) -> torch.Tensor:
    """缩放点积注意力 (Scaled Dot-Product Attention)。

    公式：softmax(Q K^T / sqrt(d_k)) V

    参数形状（... 表示任意批次维，如 batch、num_heads）：
        Q: (..., q, d_k)  查询
        K: (..., k, d_k)  键
        V: (..., k, d_v)  值（k 必须与 K 的序列长度一致）
        mask: (..., q, k) 布尔张量，True 表示"允许注意"，False 表示"屏蔽"
    返回：
        (..., q, d_v) 加权求和后的输出
    """
    # 第一步：计算相似度分数 Q K^T，形状 (..., q, k)。
    # einsum 下标只能是字母 [a-zA-Z]，不能用 _ 或数字。
    # "... q d, ... k d -> ... q k" 的含义：
    #   ...  批次维，两边同名表示对齐/广播
    #   q    查询序列维，保留到输出
    #   k    键序列维，保留到输出
    #   d    特征维，两个输入都有但输出没有 -> 沿它求和(即点积)
    # 等价于 matmul(Q, K.transpose(-2,-1))，但无需手动转置。
    # 除以 sqrt(d_k) 做缩放：防止点积随维度增大而过大，导致 softmax 过于尖锐、梯度消失。
    similarity = torch.einsum("... q d,... k d -> ... qk",Q,K) / math.sqrt(Q.shape[-1])

    # 第二步：应用 mask。mask 为 False 的位置填成 -inf，
    # 这样经过 softmax 后这些位置的权重会变成 0（被完全屏蔽）。
    # 用 ~mask 取反：True(允许) -> False(不填)，False(屏蔽) -> True(填 -inf)。
    if mask is not None:
        similarity = similarity.masked_fill(~mask, -torch.inf)

    # 第三步：沿最后一维(k 维)做 softmax，得到每个 query 对所有 key 的注意力权重，
    # 形状 (..., q, k)，每一行权重之和为 1。
    attention_weights = softmax(similarity,dim=-1)

    # 第四步：用注意力权重对 V 加权求和，形状 (..., q, d_v)。
    # "... q k, ... k v -> ... q v"：k 维只在输入出现 -> 沿它求和；
    # q、v 保留到输出。这里 v 表示 V 的特征维 d_v。
    return torch.einsum("... q k,... k v -> ...q v",attention_weights,V)


class CausalMultiHeadSelfAttention(torch.nn.Module):
    """因果多头自注意力 (Causal Multi-Head Self-Attention)。

    把 d_model 维输入拆成 num_heads 个头，每个头维度 d_head = d_model / num_heads，
    各自独立做缩放点积注意力（带因果掩码，只能看到当前及之前的 token），
    最后把各头输出拼接并经过输出投影。
    可选地，对 Q/K 施加 RoPE（旋转位置编码）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        rope_theta: float | None = None,
        max_seq_len: int | None = None,
        rope: RotaryPositionalEmbedding | None = None
    ) -> None:
        # d_model 必须能被 num_heads 整除，否则无法均分给每个头。
        assert(d_model % num_heads == 0)
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q/K/V 投影：把输入分别映射成查询、键、值（维度不变，都是 d_model）。
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # 输出投影：把拼接后的多头结果再映射回 d_model。
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 可选 RoPE：注意 d_k 用的是 d_head（每个头的维度），不是 d_model。
        # 先显式设为 None，保证属性一定存在。
        # 原因：nn.Module 的 __getattr__ 在属性不存在时会抛 AttributeError，
        # 所以 forward 里写 `if self.rope is not None` 时，属性必须已定义。
        self.rope = None
        if rope_theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_head, max_seq_len,device=device)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 (batch, seq, d_model)
        # 1) reshape 拆出 head 维 -> (batch, seq, num_heads, d_head)
        # 2) transpose 把 num_heads 换到 seq 前面 -> (batch, num_heads, seq, d_head)
        # 这样每个头就能独立做注意力（最后两维是 (seq, d_head)）。
        return x.reshape(*x.shape[:-1],self.num_heads,self.d_head).transpose(-3,-2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 (batch, num_heads, seq, d_head)
        # 1) transpose 还原顺序 -> (batch, seq, num_heads, d_head)
        # 2) reshape 合并最后两维 -> (batch, seq, d_model)
        # 注意：必须先转置再 reshape，且用转置后的 shape 计算，否则元素数不匹配。
        x = x.transpose(-3,-2)
        return x.reshape(*x.shape[:-2], self.d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 生成 (seq, seq) 的布尔下三角矩阵：mask[i, j] = (i >= j)。
        # True 表示位置 i 允许注意位置 j（j <= i），False 表示屏蔽未来位置。
        position = torch.arange(seq_len, device=device)
        return position[:, None] >= position[None, :]
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq, d_model)
        # 分别投影并拆头，得到 (batch, num_heads, seq, d_head)。
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        # 如果有 RoPE，对 q 和 k 应用 RoPE（只旋转 Q/K，不旋转 V）。
        # 此时 q/k 形状 (batch, heads, seq, d_head)，token_positions 会广播到该形状。
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            positions_for_heads = token_positions.unsqueeze(-2)
            q = self.rope(q, positions_for_heads)
            k = self.rope(k, positions_for_heads)
        # 因果掩码：位置 i 只能看到 j <= i，形状 (seq, seq)，会广播到 (batch, heads, seq, seq)。
        mask = self._causal_mask(x.shape[-2], x.device)
        # 每个头独立做缩放点积注意力 -> (batch, num_heads, seq, d_head)。
        heads = scaled_dot_product_attention(q, k, v, mask)
        # 合并各头 -> (batch, seq, d_model)，再过输出投影。
        return self.output_proj(self._merge_heads(heads))

class TransformerBlock(torch.nn.Module):
    """Transformer 块 (Pre-Norm 结构)。

    采用"先归一化再子层"的 Pre-Norm 残差结构（比 Post-Norm 更易训练）：
        h = x + Attn(RMSNorm(x))
        y = h + FFN(RMSNorm(h))
    其中 Attn 是带 RoPE 的因果多头自注意力，FFN 是 SwiGLU。
    """

    def __init__(
        self,
        d_model:int,
        num_heads:int,
        d_ff:int,
        max_seq_len:int,
        theta:float,
        device:torch.device | str | None = None,
        dtype:torch.dtype | None = None
    ) -> None:
        """
        Args:
            d_model (int): 模型隐藏维度。
            num_heads (int): 注意力头数，d_model 必须能被它整除。
            d_ff (int): 前馈网络内部维度。
            max_seq_len (int): 最大序列长度（用于 RoPE 预缓存）。
            theta (float): RoPE 基频参数。
            device (torch.device | str | None): 参数所在设备。
            dtype (torch.dtype | None): 参数数据类型。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        # 第一个 RMSNorm，作用于注意力子层之前。
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        # 因果多头自注意力，内部自带 RoPE。
        self.attn = CausalMultiHeadSelfAttention(d_model,num_heads,device=device,dtype=dtype,rope_theta=theta,max_seq_len=max_seq_len)
        # 第二个 RMSNorm，作用于前馈子层之前。
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
        # SwiGLU 前馈网络。
        self.ffn = SwiGLU(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        """前向传播：两个 Pre-Norm 残差子层。

        Args:
            x (torch.Tensor): 输入，形状 (batch, seq_len, d_model)。

        Returns:
            torch.Tensor: 输出，形状 (batch, seq_len, d_model)。
        """
        # 注意力子层：先归一化，再注意力，最后残差相加。
        h = self.attn(self.ln1(x)) + x
        # 前馈子层：先归一化，再 FFN，最后残差相加。
        return self.ffn(self.ln2(h)) + h


class TransformerLM(torch.nn.Module):
    """Transformer 语言模型 (Decoder-only)。

    结构：
        token embedding -> N 个 TransformerBlock -> 最终 RMSNorm -> lm_head 投影到词表
    输出每个位置对下一个 token 的未归一化 logits（形状 (batch, seq, vocab_size)）。
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Args:
            vocab_size (int): 词表大小。
            context_length (int): 最大上下文长度（序列长度上限）。
            d_model (int): 模型隐藏维度。
            num_layers (int): Transformer 块的层数。
            num_heads (int): 每层注意力头数。
            d_ff (int): 前馈网络内部维度。
            rope_theta (float): RoPE 基频参数。
            device (torch.device | str | None): 参数所在设备。
            dtype (torch.dtype | None): 参数数据类型。
        """
        super().__init__()
        # 词嵌入：把 token id 映射成 d_model 维向量。
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # 堆叠 num_layers 个 Transformer 块（ModuleList 会正确注册子模块）。
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        # 最后一层归一化。
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # 输出投影：把 d_model 映射到词表大小，得到每个 token 的 logits。
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：token id -> 下一个 token 的 logits。

        Args:
            x (torch.Tensor): 输入 token id，形状 (batch, seq_len)。

        Returns:
            torch.Tensor: logits，形状 (batch, seq_len, vocab_size)。
        """
        # 1) 词嵌入：(batch, seq) -> (batch, seq, d_model)
        x = self.token_embeddings(x)
        # 2) 逐层通过 Transformer 块。
        for layer in self.layers:
            x = layer(x)
        # 3) 最终归一化。
        x = self.ln_final(x)
        # 4) 投影到词表，得到 logits。
        return self.lm_head(x)  

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """交叉熵损失 (数值稳定版)。

    对每个样本计算 -log softmax(logits)[target]，再对 batch 取平均。
    实现上先减去每行最大值，避免 exp 溢出。

    Args:
        logits (torch.Tensor): 未归一化的预测分数，形状 (batch_size, vocab_size)。
        targets (torch.Tensor): 正确类别的索引，形状 (batch_size,)，取值 [0, vocab_size)。

    Returns:
        torch.Tensor: 标量损失（batch 上的平均交叉熵）。
    """
    # 数值稳定：先减去每行的最大值，避免 exp 溢出（logits 可能很大，如 1000+）
    logits = logits - torch.max(logits, dim=1, keepdim=True).values
    # log-sum-exp：log(Σ_j exp(z_j))，此时 z 已减过 max，不会溢出
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=1))
    # 取出每个样本目标类别的 logit：z_y
    # 注意：gather 的 index 必须和 input 维度数相同，所以用 unsqueeze(-1) 变成 (N, 1)，
    # 不能 squeeze 掉，否则维度不匹配会报 "Index tensor must have the same number of dimensions"
    target_logits = torch.gather(logits, 1, targets.unsqueeze(-1)).squeeze(-1)
    # 交叉熵 = log-sum-exp - z_y（这里 logits 已经减过 max，不要再减一次）
    loss = log_sum_exp - target_logits
    return loss.mean()

def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """按全局 L2 范数裁剪梯度 (gradient clipping)。

    把所有参数的梯度拼成一个整体向量，计算其全局 L2 范数；
    若超过 max_l2_norm，则按比例缩放所有梯度，使其范数等于 max_l2_norm。
    这是防止梯度爆炸的常用手段。梯度是原地(in-place)修改的。

    Args:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数集合。
        max_l2_norm (float): 允许的最大全局 L2 范数。

    Returns:
        None: 直接原地修改各参数的 .grad。
    """
    # 收集所有非空梯度（有些参数可能没有梯度）。
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    # 全局 L2 范数 = sqrt(所有梯度元素平方和)。detach 避免把裁剪操作计入计算图。
    global_norm = torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads))
    # 只有超过阈值才裁剪；缩放系数 = max_l2_norm / (global_norm + eps)，eps 防止除零。
    if global_norm > max_l2_norm:
        with torch.no_grad():
            for g in grads:
                g.mul_(max_l2_norm / (global_norm + 1e-6))