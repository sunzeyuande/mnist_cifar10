# 手写数字识别
import pickle
import gzip
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from torchstat import stat

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # gpu加速

with gzip.open('data/mnist/mnist.pkl_3.gz', "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")  # 加载数据集，划分训练集验证集及其对应标签

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))  # 将数据转为张量格式
n, c = x_train.shape  # 获得张量形状
bs = 100  # batch_size大小
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)  # 划分训练集并打乱顺序
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)  # 划分验证集


class mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=0)  # 定义第一个卷积层
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=0)  # 定义第二个卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)  # 定义第三个卷积层
        self.fc1 = nn.Linear(128*1*1, 10)  # 定义全连接层

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)  # 将数据集中存储的784*1图像还原成28*28形状
        xb = F.relu(self.conv1(xb))  # 经过第一个卷积层并用relu激活
        xb = F.max_pool2d(xb, 2, stride=2)  # 最大池化
        xb = F.relu(self.conv2(xb))  # 经过第二个卷积层并用relu激活
        xb = F.max_pool2d(xb, 2, stride=2)  # 最大池化
        xb = F.relu(self.conv3(xb))  # 经过第三个卷积层并用relu激活
        xb = F.max_pool2d(xb, 2, stride=2)  # 最大池化
        xb = xb.view(-1, 128*1*1)  # 将张量变为一维
        xb = self.fc1(xb)  # 全连接层进行分类
        return xb


# device = torch.device('cpu')
# stat(mnist(), (1, 784, 100))  # 计算网络参数
model = mnist().to(device)  # 创建一个网络实例
loss_func = F.cross_entropy  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters())  # adam优化器

lr = 0.05  # 学习率
epochs = 20  # 训练轮数
loss_plot = []  # 画图采样点

train_start = time.time()

for epoch in range(epochs):
    total_loss = 0
    plot_loss = 0
    count = 0
    train_correct = 0
    for xb, yb in train_dl:
        xb = xb.to(device)  # 将张量部署到gpu
        yb = yb.to(device)

        pred = model(xb)  # 预测值
        loss = loss_func(pred, yb)  # 计算损失值

        loss.backward()  # 更新权重
        optimizer.step()  # 优化器梯度更新
        optimizer.zero_grad()  # 优化器梯度清零

        preds = torch.argmax(pred, dim=1)  # 识别数字是几
        train_correct += sum(preds == yb)

        count += 1
        total_loss += loss.item()
        plot_loss += loss.item()
        if count % 10 == 0:
            # print('train_loss:', plot_loss / 10)
            loss_plot.append(plot_loss/10)
            plot_loss = 0
    train_acc = train_correct.item()/50000  # 计算acc
    print('epoch:', epoch+1, 'train_loss:', total_loss / 500, 'train_acc:', train_acc)

    plt.figure()  # 每轮训练后绘图
    plt.plot(loss_plot)
    plt.title('loss of epoch'+str(epoch+1))
    plt.xlabel('number of 10batches')
    plt.ylabel('loss')
    plt.show()
train_finish = time.time()
print('训练时间(s)：', train_finish-train_start)  # 计算训练时间
torch.save(model.state_dict(), './data/mnist/mnist.pkl')  # 保存网络参数

with torch.no_grad():  # 网络权重不更新
    lst = []
    loss = 0
    test_correct = 0
    for xb, yb in valid_dl:
        xb = xb.to(device)  # 将张量部署到gpu
        yb = yb.to(device)
        out = model(xb)  # 输入经过网络输出
        loss_pre = loss_func(out, yb).item()  # 计算loss
        loss += loss_pre
        preds = torch.argmax(out, dim=1)  # 识别数字是几
        lst.append(yb.cpu().numpy())
        lst.append(preds.cpu().numpy())
        test_correct += (sum(preds == yb)).item()
    test_acc = test_correct/10000  # 计算acc
    test_loss = loss/50
    print('实际数字：', lst[-2][0:30])
    print('识别数字：', lst[-1][0:30])
    print('test_loss:', test_loss, 'test_acc:', test_acc)
