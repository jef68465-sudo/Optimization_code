from typing import List
import torch

class SafetyClassifier(torch.nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

def train_classifier(model: SafetyClassifier, features: torch.Tensor, labels: torch.Tensor, epochs: int = 3, lr: float = 1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    for e in range(epochs):
        logits = model(features)
        loss = crit(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model