import time
import wandb

# 1. 初始化（如果国内网络连不上，加上 mode="offline"）
wandb.init(
    project="diy-llm-test",
    name="hello-wandb",
    config={"lr": 1e-4, "batch_size": 32}
)

# 2. 模拟 5 步训练，把假 loss 传上去
for step in range(5):
    fake_loss = 1.0 / (step + 1)
    wandb.log({"train/loss": fake_loss, "step": step})
    time.sleep(0.5)

wandb.finish()