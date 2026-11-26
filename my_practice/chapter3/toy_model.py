import torch
import torch.nn as nn

import wandb

# ==========================================
# 0. 设备自适应（兼容你的 Intel Mac CPU 和以后云端 GPU）
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)  # 固定随机种子，保证每次结果可复现
print(f"使用设备: {device}")

wandb.init(project="diy-llm-ch3-toy")

# ==========================================
# 1. 定义 Toy Model（继承 nn.Module，第 4 章也完全这么写！）
# ==========================================
class ToyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # 定义两层线性层 + 激活函数
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 实例化模型并搬到指定设备上
model = ToyMLP(in_dim=3, hidden_dim=64, out_dim=1).to(device)

# 数一下第 3 章学的参数量
num_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {num_params} 个 (FP32 静态权重约占 {num_params * 4 / 1024:.2f} KB)")

# ==========================================
# 2. 准备模拟数据集（500 条样本，每条 3 个特征）
# ==========================================
X = torch.randn(500, 3, device=device)
# 真实规律：y = 2*x0 - 3*x1 + x2^2（模型不知道这个公式！）
Y = (2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] ** 2).unsqueeze(1)

# 切分 400 条当训练集，100 条当验证集（期末考，不参与训练）
X_train, Y_train = X[:400], Y[:400]
X_val, Y_val = X[400:], Y[400:]

# ==========================================
# 3. 定义损失函数 & AdamW 优化器
# ==========================================
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

# ==========================================
# 4. 训练循环（五步曲）
# ==========================================
epochs = 200
for epoch in range(1, epochs + 1):
    model.train()  # 切换到训练模式

    optimizer.zero_grad()          # ① 清空旧梯度
    pred = model(X_train)          # ② 前向传播
    loss = criterion(pred, Y_train)# ③ 计算误差
    loss.backward()                # ④ 反向传播算梯度
    optimizer.step()               # ⑤ AdamW 更新参数

    log_dict = {"train_loss": loss.item()}
    # 每 20 轮做一次验证集考试（记住第 3 章重点：必须包 torch.no_grad() 省内存！）
    if epoch % 20 == 0 or epoch == 1:
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val)
        log_dict["val_loss"] = val_loss.item()
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        wandb.log(log_dict, step=epoch)

# ==========================================
# 5. 推理测试：喂一个它没见过的新样本看看猜得准不准
# ==========================================
model.eval()
with torch.no_grad():
    test_x = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    # 真实答案应该是: 2*1 - 3*2 + 3^2 = 2 - 6 + 9 = 5.0
    guess = model(test_x)
    print(f"\n测试输入 [1, 2, 3] | 真实答案: 5.0000 | 模型预测: {guess.item():.4f}")

wandb.finish()