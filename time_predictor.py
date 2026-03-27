import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32, device=device)
y = torch.tensor([[4.05], [3.93], [3.65], [4.01], [8.37], [308.12], [86879.50], [6992707.50]], dtype=torch.float32, device=device)
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
    test_val = torch.tensor([[10.0]], device=device)
    log_pred = model(test_val)
    pred = torch.exp(log_pred)
    print(f"Prediction for x=10: {pred.item()} ms")
    test_val = torch.tensor([[8.0]], device=device)
    log_pred = model(test_val)
    pred = torch.exp(log_pred)
    print(f"Prediction for x=8: {pred.item()} ms")


x1, x2 = torch.tensor([[1.0]], device=device), torch.tensor([[8.0]], device=device)
y1 = torch.exp(model(x1)).item()
y2 = torch.exp(model(x2)).item()

B = np.log(y2 / y1) / (8.0 - 1.0)
A = y1 / np.exp(B * 1.0)

print(f"The model's behavior approximates: y = {A:.2f} * e^({B:.4f} * x)")