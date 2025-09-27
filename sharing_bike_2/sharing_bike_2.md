# 共享单车使用量预测
## 任务二：拟合非线性关系
### 代码结构
1. 定义神经网络模型：
```python
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
```
* 网络结构：3层全连接神经网络
* 激活函数：使用ReLU激活函数
2. 模型训练：
```python
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
```
* 使用Adam优化器
* 在整个训练集上进行批量梯度下降
* 保存每轮损失值
* 每500轮输出一次损失值
3. 结果分析与可视化
```python
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
```
* 评估模型泛化能力
* 输出拟合曲线，损失曲线
### 预期输出
* 训练集、测试集大小
* 每轮epoch
* 测试集损失
* 原始数据分布以及非线性拟合曲线
* 训练损失曲线
### 思考题
* Q:定义模型时，只使用线性层和激活函数层，且激活函数层只使用ReLU。你观察到什么现象？
* A:出现较明显的拐点（其实看上去也还好？）
* Q:为什么会产生这种现象？
* A:relu函数在0处不可导，输出的是多个分段线性函数的组合
* Q:假设这种现象不符合预期，应该怎么避免？
* A:使用平滑激活函数tanh，并进行数据标准化