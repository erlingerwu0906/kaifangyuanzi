import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'
# 1. 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. 构建你的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # --- 在这里定义你的网络层 ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # --- 在这里定义前向传播逻辑 ---
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # 展平张量
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = SimpleCNN().to(device)

# --- 请你完成这部分代码 ---
# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 4. 训练和评估模型
epoch_num = 10

train_loss_values = []
train_acc_values = []
test_acc_values = []

for epoch in range(epoch_num):
    running_loss = 0.0
    correct = 0
    total = 0
    
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct / total
    train_loss_values.append(train_loss)
    train_acc_values.append(train_acc)

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    test_acc_values.append(test_acc)
    
    print(f'Epoch {epoch + 1}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%')

print('训练完成')

# 5. 可视化
# --- Loss和Accuracy曲线 ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epoch_num + 1), train_loss_values, marker='o', color='blue')
plt.title('训练过程中的损失变化')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epoch_num + 1), train_acc_values, marker='o', color='blue', label='训练准确率')
plt.plot(range(1, epoch_num + 1), test_acc_values, marker='s', color='red', label='测试准确率')
plt.title('训练过程中的准确率变化')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            if len(c.shape) > 0:
                class_correct[label] += c[i].item()
            else:
                class_correct[label] += c.item()
            class_total[label] += 1

print('\n各类别准确率:')
for i in range(10):
    print(f'{classes[i]}的准确率: {100 * class_correct[i] / class_total[i]:.2f}%')