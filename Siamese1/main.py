import os
import random
import linecache
import numpy as np
import PIL.ImageOps
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):

    def __init__(self, dataDir_txt, transform=None, target_transform=None, should_invert=False):
        self.dataDir_txt = dataDir_txt
        self.transform = transform#对图像进行转换操作，如数据增强
        self.target_transform = target_transform#对目标标签进行转换操作
        self.should_invert = should_invert#是否反转图像

    #用于获取索引index处的样本
    def __getitem__(self, index):
        line1 = linecache.getline(self.dataDir_txt, random.randint(1, self.__len__()))#从文本文件中随机选择一行
        line1.strip('\n')#移除字符串开头和结尾的换行符
        img0_list = line1.split()#如:img0_list=['att_faces/s38/4.pgm', '37']

        line2=linecache.getline(self.dataDir_txt, random.randint(1, self.__len__()))
        line2.strip('\n')
        img1_list = line2.split()

        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                if img0_list[1] == img1_list[1]:
                    break
                line2=linecache.getline(self.dataDir_txt, random.randint(1, self.__len__()))
                line2.strip('\n')
                img1_list = line2.split()

        img0 = Image.open(img0_list[0])
        img1 = Image.open(img1_list[0])
        img0 = img0.convert("L")#转换为灰度模式（L）
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        #相同标0，不同标1
        return img0, img1, torch.from_numpy(np.array([int(img0_list[1] != img1_list[1])], dtype=np.float32))

    def __len__(self):
        f = open(self.dataDir_txt, 'r')
        num = len(f.readlines())
        f.close()
        return num

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #卷积层
        self.cnn = nn.Sequential(
            # 在输入的边界周围填充一个像素来进行反射填充，填充的大小为 1
            nn.ReflectionPad2d(padding=1),
            # 卷积层，输入通道数为 1，输出通道数为 4，卷积核大小为 3x3
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
            # ReLU激活函数，inplace=True 表示将计算结果直接存储在输入张量中，节省内存
            nn.ReLU(inplace=True),
            # 二维批归一化，规范卷积层的输出，加速收敛. 4表示要规范化的特征图的通道数。
            nn.BatchNorm2d(num_features=4),
            # 二维 dropout，以 0.2 的概率随机丢弃一部分特征图，用于减少过拟合
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(p=.2),
        )

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    # 单输入的前向传播过程，将输入通过CNN和FC进行计算并返回结果
    def forward_single(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    # 整个Siamese的前向传播，接收两个输入(input1,input2),分别经过两个相同的前向传播路径
    def forward(self, input1, input2):
        output1 = self.forward_single(input1)
        output2 = self.forward_single(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def plot_loss_curve(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

def generate_imgDirs_labels():
    folder_path=train_data_root
    txt_path=train_data_txt_root
    with open(txt_path, 'w') as f:
        subfolders = sorted([folder for folder in os.listdir(folder_path)])

        for i, folder in enumerate(subfolders):
            subfolder_path = os.path.join(folder_path, folder)
            
            if not os.path.isdir(subfolder_path):
                continue

            image_names = sorted(os.listdir(subfolder_path))

            for image_name in image_names:
                if image_name == "Thumbs.db":
                    continue

                image_path = os.path.join(subfolder_path, image_name)
                f.write(f"{image_path} {i}\n")

train_data_root = 'D:/Code/FaceRecognizer/pic'
train_data_txt_root = 'train_imgsDir.txt'
train_batch_size, train_epochs = 32, 30

generate_imgDirs_labels()  # Generate image path and his/her name
train_data = MyDataset(dataDir_txt=train_data_txt_root,
                       transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]),
                       should_invert=False)
train_dataloader = DataLoader(dataset=train_data,
                              shuffle=True,
                              num_workers=2,
                              batch_size=train_batch_size)

net = SiameseNetwork().cpu()
loss = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

epoch_history = []
loss_history = []
iteration = 0
total_batches = len(train_dataloader)

if __name__ == '__main__':
    for epoch in range(0, train_epochs):
        total_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = Variable(img0).cpu(), Variable(img1).cpu(), Variable(label).cpu()
            
            output1, output2 = net(img0, img1)

            optimizer.zero_grad()
            loss_contrastive = loss(output1, output2, label)
            loss_contrastive.backward() # BP
            optimizer.step() # update weight

            total_loss += loss_contrastive.item()

        iteration+=1
        avg_loss = total_loss / total_batches

        epoch_history.append(iteration)
        loss_history.append(avg_loss)

        print("Epoch:{},   loss{}\n".format(epoch, avg_loss))

    plot_loss_curve(epoch_history, loss_history)
    torch.save(net.state_dict(), 'SiameseNet.pth')
