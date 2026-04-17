import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

x = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32, device=device)
y = torch.tensor([[0.0036], [0.0041], [0.0038], [0.0043], [0.0059], [0.1164], [8.9226], [637.8328]], dtype=torch.float32, device=device)
y_log = torch.log(y)

dataset = TensorDataset(x, y_log)
loader = DataLoader(dataset, shuffle=True)

class PwdModel(nn.Module):
    def __init__(self):
        super(PwdModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = PwdModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 2000

loop = tqdm(range(epochs), leave=True)

for epoch in loop:
    for inputs, labels in loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        loop.set_postfix_str(f"Loss = {loss.item():.2f}")

print(loss.item())
torch.save(model, "model.pt")

model.eval()
with torch.no_grad():
    test_val = torch.tensor([[8.0]], device=device)
    log_pred = model(test_val)
    pred = torch.exp(log_pred)
    print(f"Prediction for x=8: {pred.item()} seconds")
    test_val = torch.tensor([[9.0]], device=device)
    log_pred = model(test_val)
    pred = torch.exp(log_pred)
    print(f"Prediction for x=9: {pred.item()} seconds")
    test_val = torch.tensor([[10.0]], device=device)
    log_pred = model(test_val)
    pred = torch.exp(log_pred)
    print(f"Prediction for x=10: {pred.item()} seconds")


x1, x2 = torch.tensor([[1.0]], device=device), torch.tensor([[8.0]], device=device)
y1 = torch.exp(model(x1)).item()
y2 = torch.exp(model(x2)).item()

B = np.log(y2 / y1) / (8.0 - 1.0)
A = y1 / np.exp(B * 1.0)

print(f"The model's behavior approximates: y = {A:.4f} * e^({B:.4f} * x)")

x_axis = []
y_axis = []

with torch.no_grad():
    for i in range(20):
        x_axis.append(i)
        t = torch.tensor([[i]], dtype=torch.float32, device=device)
        out = torch.exp(model(t)).cpu().numpy().item()
        print(i, out)
        y_axis.append(out)

plt.plot(x_axis, y_axis)
plt.xlabel("password length")
plt.ylabel("time in seconds")
plt.savefig("lvtrelation.png")
plt.show()