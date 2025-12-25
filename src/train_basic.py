import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1) Transform ---
transform = transforms.ToTensor()

# --- 2) Dataset ---
train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform
)

# --- 3) DataLoader ---
train_loader = DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True
)

# --- 4) Model ---
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# --- 5) Loss & Optimizer ---
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# --- 6) Training loop ---
for epoch in range(1):        # 1 epoch for now
    for images, labels in train_loader:
        # forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch finished. Last batch loss:", loss.item())

# --- 7) Save the model ---
# --- Save model ---
torch.save(model.state_dict(), "basic_cifar_model.pth")
print("Model saved!")