import torch
import torch.nn as nn
import numpy as np

# ===============================
# 1) تجهيز البيانات
# ===============================
# درجات حرارة بسيطة كمثال
temperatures = np.array([30, 31, 32, 33, 34, 35, 36, 37, 38, 39], dtype=float)
temperatures = temperatures / 100.0   # تطبيع البيانات (Normalization)

time_steps = 3
X, y = [], []

for i in range(len(temperatures) - time_steps):
    X.append(temperatures[i:i+time_steps])
    y.append(temperatures[i+time_steps])

X = np.array(X)
y = np.array(y)

# تحويل إلى Tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (batch, time_step, feature)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# ===============================
# 2) بناء نموذج LSTM
# ===============================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h, c) = self.lstm(x)  # out: (batch, time_step, hidden_size)
        out = self.fc(out[:, -1, :])  # نأخذ آخر خطوة زمنية
        return out

model = LSTMModel()

# ===============================
# 3) تهيئة الخسارة والمحول
# ===============================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ===============================
# 4) تدريب النموذج
# ===============================
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# ===============================
# 5) اختبار النموذج
# ===============================
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[0.36, 0.37, 0.38]], dtype=torch.float32).unsqueeze(-1)
    predicted = model(test_input)
    print("\nPrediction:", predicted.item() * 100, "°C")
