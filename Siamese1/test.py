import os
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #卷积层
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=4),
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

def predict(dataset_path):
    model = SiameseNetwork()
    model.load_state_dict(torch.load('SiameseNet.pth'))
    correct_num=0
    subfolders = sorted([folder for folder in os.listdir(dataset_path)])
    total_folders = len(subfolders)

    for i, folder in enumerate(subfolders):
        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        image_names = sorted(os.listdir(folder_path))
        if len(image_names) != 2:
            continue

        image_paths = [os.path.join(folder_path, image_name)
                       for image_name in image_names]
        
        similarity_score = test_model(model, image_paths[0], image_paths[1])

        if (int(folder[0])==0 and similarity_score<0.6) or (int(folder[0])==1 and similarity_score>=0.6):
            correct_num+=1
        
        print_progress_bar(i + 1, total_folders, prefix='Testing:')

    accuracy= correct_num / total_folders
    print('accuracy=',accuracy)

def test_model(model, image1_path, image2_path):
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

    image1 = Image.open(image1_path).convert("L")
    image1 = transform(image1).unsqueeze(0)

    image2 = Image.open(image2_path).convert("L")
    image2 = transform(image2).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output1, output2 = model(image1, image2)

    euclidean_distance = F.pairwise_distance(output1, output2)
    similarity_score = 1 - euclidean_distance.item()

    return similarity_score

def print_progress_bar(iteration, total, prefix='', suffix='', length=70, fill='█'):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

if __name__ == '__main__':
    predict('D:/Code/FaceRecognizer/test')
    # accuracy= 0.5442508710801394