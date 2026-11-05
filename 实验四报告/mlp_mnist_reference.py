# -*- coding: utf-8 -*-
"""MNIST version based on the uploaded Experiment 4 PDF. Requires internet for first download."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=False)

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))

model = mlp().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1, 31):
    model.train()
    for x, y in train_loader:
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
    model.eval()
    correct, test_loss, test_batches = 0, 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.view(x.shape[0], -1).to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y).item()
            correct += (output.argmax(dim=1) == y).sum().item()
            test_batches += 1
    print(f'epoch:{epoch}, loss:{test_loss/test_batches:.4f}, acc:{correct/len(test_loader.dataset):.4f}')
