import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- transforms ---
transform = transforms.ToTensor()

# --- test dataset ---
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False
)

# --- model (same structure as training) ---
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# --- load trained weights ---
model.load_state_dict(torch.load("basic_cifar_model.pth"))

# --- evaluation ---
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")