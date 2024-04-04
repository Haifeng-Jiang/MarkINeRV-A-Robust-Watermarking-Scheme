import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from your_dataset import YourDataset  # 实现一个你自己定义的 data loader

import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义超参数
batch_size = 4

# 加载并处理每张图片
images = []
for i in range(13):
    img = Image.open(f"your_directory/{i}.jpg")
    transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
    img = transform(img)
    images.append(img)

# 将图片组成一个 batched Tensor
inputs = torch.stack(images, dim=0)

# 加载恢复模型
model = torch.load('your_model_path')

# 将 batched Tensor 输入到模型中获取重建的输出 Tensor
restored_images = model(inputs)

# 在进行进一步操作之前，可能需要将重建的输出 Tensor 转换回 PIL 图像格式。
# 定义超参数
batch_size = 4
num_epochs = 10
learning_rate = 0.001

# 定义数据集
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = YourDataset(transform=transform)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# 定义模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        return outputs


# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for images in data_loader:
        model.zero_grad()
        reconstructed_images = model(images)
        loss = criterion(reconstructed_images, images)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


inputs = torch.randn(13, 3, 256, 256)
restored_images = model(inputs)