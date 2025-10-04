import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# 1. 加载数据
data = pd.read_csv('day.csv')
X_df = data[['temp']]
y_df = data[['cnt']]

# 2. 转换为PyTorch张量
X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
y_tensor = torch.tensor(y_df.values, dtype=torch.float32)

# 检查数据维度：X应为 (样本数, 1), y应为 (样本数, 1)
print(X_tensor.shape, y_tensor.shape)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.scatter(X_tensor.numpy(), y_tensor.numpy(), alpha=0.5, color='b', s=20)
plt.xlabel('温度 ')
plt.ylabel('自行车日租用量')
plt.title('温度与自行车日租用量的关系')
plt.grid(True, alpha=0.3)
plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
losses = []
for epoch in range(epochs):
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

weight = model.linear.weight.item()
bias = model.linear.bias.item()
print(f"\n结果: y = {weight:.4f} * x + {bias:.4f}")

plt.figure(figsize=[15, 5])
plt.subplot(1, 2, 1)
plt.scatter(X_tensor.numpy(), y_tensor.numpy(), alpha=0.4, color='b', label='原始数据')

x_range = torch.linspace(X_tensor.min(), X_tensor.max(), 100)
y_pred = model(x_range.unsqueeze(1)).detach().numpy()
plt.plot(x_range.numpy(), y_pred, color='r', linewidth=2,
         label=f'拟合直线: y = {weight:.2f}x + {bias:.2f}')

plt.xlabel('温度')
plt.ylabel('自行车日租量')
plt.title('线性回归拟合')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(epochs), losses, color='b')
plt.xlabel('训练数据循环次数')
plt.ylabel('损失')
plt.title('训练损失曲线')
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()