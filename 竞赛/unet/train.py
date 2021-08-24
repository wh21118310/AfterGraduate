from torch.autograd import Variable

from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential

def train_net(net, device, data_path, epochs=40, batch_size=8, lr=0.00001):
    # 加载训练集
    global loss
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = DataLoader(dataset=isbi_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    # 定义RMSprop算法,优化算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss().cuda()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        print(epoch)
        # 按照batch_size开始训练
        for image, label in train_loader:
            # 将数据拷贝到device中
            #解决问题二
            # image = Variable(image.to(device=device, dtype=torch.float16),requires_grad=True)
            # label = Variable(label.to(device=device, dtype=torch.float16),requires_grad=True)
            image = image.to(device=device,dtype=torch.float32)
            label = label.to(device=device,dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)

            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 清空梯度
            optimizer.zero_grad()
            # 更新参数
            loss.backward() #计算梯度
            optimizer.step() # 反向传播，更新网络参数
            loss = loss.cpu()
            print("loss :",float(loss))


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.device_count())
    # device = torch.device('cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # net.initialize_weights()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "./sardata/train/"
    train_net(net, device, data_path)