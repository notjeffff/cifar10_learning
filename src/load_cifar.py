from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
transform = transforms.ToTensor()
train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=False,
    transform=transform
)

image , label = train_data[0]
print("Image shape:", image.shape)
print("Label index:", label)

# 4) Show image
plt.imshow(image.permute(1, 2, 0))   # convert back to H,W,C for display
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

#Task 2 

from torch.utils.data import DataLoader

train_loader = DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True
    )
images, labels = next(iter(train_loader))

print("Batch shape:", images.shape)
print("Labels shape:", labels.shape)