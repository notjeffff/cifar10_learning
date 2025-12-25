import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) Transform
transform = transforms.ToTensor()

# 2) Dataset (already downloaded)
train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform
)

# 3) DataLoader
train_loader = DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True
)

# 4) Simple model (same as before)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# 5) Get one batch
images, labels = next(iter(train_loader))

print("Batch shape:", images.shape)

# 6) Forward pass
outputs = model(images)

print("Output shape:", outputs.shape)
from torch import optim

# Loss function (how wrong the model is)
loss_fn = nn.CrossEntropyLoss()

# Optimizer (how model learns)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Calculate loss for this batch
loss = loss_fn(outputs, labels)

print("Loss:", loss.item())