import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

batch_size = 10
epoch = 25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 归一化并按通道进行标准化（减去均值，再除以方差）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):  # 归一化标准化逆变换，显示图像
    img = img.to(torch.device('cpu'))
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 打印一些图像
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(labels)
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


class cifar10(nn.Module):
    def __init__(self):
        super(cifar10, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512*2*2, 10)

    def forward(self, x):
        # x = self.conv1(x)
        # x = F.relu(self.pool(x))
        # x = self.conv2(x)
        # x = self.avgpool(F.relu(x))
        # x = self.conv3(x)
        # x = self.avgpool(F.relu(x))
        # # print(x.shape)
        # x = x.view(-1, 256*1*1)
        # x = self.fc1(x)
        x = F.relu(self.conv1(x))  # 第一个卷积层，relu激活
        x = self.pool(x)  # 最大池化
        x = F.relu(self.conv2(x))  # 第二个卷积层，relu激活
        x = self.pool(x)  # 最大池化
        x = F.relu(self.conv3(x))  # 第三个卷积层，relu激活
        x = self.avgpool(x)  # 平均池化
        x = F.relu(self.conv4(x))  # 第四个卷积层，relu激活
        x = self.avgpool(x)  # 平均池化
        # print(x.shape)
        x = x.view(-1, 512*2*2)
        x = self.fc1(x)  # 全连接层分类
        return x


net = cifar10().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(net.parameters())

time1 = time.time()

for epo in range(epoch):  # 多轮迭代
    running_loss = 0.0
    for j, data in enumerate(trainloader):
        inputs, labels = data  # 测试图像以及对应的标签
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = net(inputs)  # 图像经过网络得到输出
        loss = criterion(outputs, labels)
        loss.backward()  # 网络权重更新
        optimizer.step()  # 优化器参数更新
        optimizer.zero_grad()  # 优化器梯度清零

        # print statistics
        running_loss += loss.item()
        if j % 100 == 99:    # 每100batch打印信息
            print('[%d, %5d] loss: %.3f' % (epo+1, j+1, running_loss/100))
            running_loss = 0.0

time2 = time.time()
print('Finished Training, Time Using(s):', time2-time1)

dataiter = iter(testloader)
images, labels = dataiter.next()  # 测试图像及其对应标签
images = images.to(device)
labels = labels.to(device)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

correct = 0
total = 0
with torch.no_grad():  # 网络权重不更新
    for data in testloader:
        images, labels = data  # 测试图像及其对应标签
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)  # 输入测试图像得到预测分类
        preds = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += sum(preds == labels)

print('Accuracy of the network on the test images: %d %%' % (100*correct/total))  # 计算整体的分类acc

class_correct = list(0. for i in range(10))  # 流程同上，分别计算不同类的分类acc
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        preds = torch.argmax(outputs, dim=1)  # 识别数字是几
        c = (preds == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100*class_correct[i]/class_total[i]))
