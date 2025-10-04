import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
torch.manual_seed(42)
np.random.seed(42)
n_samples = 200
temp = np.random.uniform(10, 35, n_samples)
cnt = 800 + 60 * temp + 2 * (temp - 22) ** 2 + np.random.normal(0, 150, n_samples)

X_tensor = torch.tensor(temp, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(cnt, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

class NonLinearModel(nn.Module):
    def __init__(self, hidden_size=64):
        super(NonLinearModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


    def forward(self, x):
        return self.network(x)


model = NonLinearModel(hidden_size=64)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 2500
losses = []

for epoch in range(epochs):
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

temp_range = torch.linspace(X_tensor.min(), X_tensor.max(), 100).view(-1, 1)

model.eval()
with torch.no_grad():
    predictions_range = model(temp_range)
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)

print(f"测试集损失: {test_loss.item():.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_tensor.numpy(), y_tensor.numpy(), alpha=0.5, color='blue', s=20, label='原始数据')
plt.plot(temp_range.numpy(), predictions_range.numpy(), color='red', linewidth=2, label='神经网络拟合曲线')
plt.xlabel('温度')
plt.ylabel('自行车日租用量')
plt.title('非线性拟合结果')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(epochs), losses, color='green')
plt.xlabel('训练轮次(Epochs)')
plt.ylabel('损失 ')
plt.title('训练损失曲线')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()