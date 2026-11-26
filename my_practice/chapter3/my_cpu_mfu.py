import time
import torch

# 1. 看看 PyTorch 当前用了多少个 CPU 线程
print(f"当前 PyTorch CPU 线程数: {torch.get_num_threads()}")

# ==========================================
# 实验 A：测纯矩阵乘法 (GEMM) 的实际算力
# ==========================================
N = 2048
A = torch.randn(N, N, dtype=torch.float32)
B = torch.randn(N, N, dtype=torch.float32)

# 先热身 3 次（防止首次调用分配内存、初始化线程池拖慢计时）
for _ in range(3):
    _ = A @ B

iters = 10
t0 = time.perf_counter()
for _ in range(iters):
    C = A @ B
t1 = time.perf_counter()

elapsed = (t1 - t0) / iters
# 两个 N×N 矩阵相乘，浮点运算次数 = 2 * N^3
gemm_flops = 2 * (N ** 3)
gemm_gflops = gemm_flops / elapsed / 1e9
print(f"[纯矩阵乘法] 单次耗时: {elapsed*1000:.2f} ms | 实际算力: {gemm_gflops:.2f} GFLOPS")


# ==========================================
# 实验 B：用 6ND 公式测“真实训练一步”的模型有效算力 (用于算 MFU)
# ==========================================
# 构造一个简易线性网络模拟训练一步
in_dim, hidden_dim, out_dim = 1024, 2048, 1024
batch_tokens = 512  # 相当于 Batch_Size * Seq_Len = 512 个 Token

model = torch.nn.Sequential(
    torch.nn.Linear(in_dim, hidden_dim, bias=False),
    torch.nn.GELU(),
    torch.nn.Linear(hidden_dim, out_dim, bias=False),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(batch_tokens, in_dim)
target = torch.randn(batch_tokens, out_dim)

# 计算模型参数量 P
num_params = sum(p.numel() for p in model.parameters())
# 根据第 3 章 6ND 公式，训练一步的理论模型计算量 FLOPs ≈ 6 * 参数量 * Token数
step_model_flops = 6 * num_params * batch_tokens

# 热身 3 步
for _ in range(3):
    optimizer.zero_grad()
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    optimizer.step()

# 正式计时 10 步
train_iters = 10
t0 = time.perf_counter()
for _ in range(train_iters):
    optimizer.zero_grad()
    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    optimizer.step()
t1 = time.perf_counter()

step_time = (t1 - t0) / train_iters
actual_model_gflops = step_model_flops / step_time / 1e9
print(f"[模拟训练] 参数量: {num_params/1e6:.2f}M | 单步耗时: {step_time*1000:.2f} ms | 有效模型算力: {actual_model_gflops:.2f} GFLOPS")

# ==========================================
# 计算 MFU：把下面换成你这台 Intel CPU 的理论峰值 (GFLOPS)
# ==========================================
# 先用 sysctl 查好再填进来，这里先假设比如 300 GFLOPS
PEAK_CPU_GFLOPS = 499.2
mfu = (actual_model_gflops / PEAK_CPU_GFLOPS) * 100
print(f"假设理论峰值为 {PEAK_CPU_GFLOPS} GFLOPS 时，训练 MFU ≈ {mfu:.2f}%")