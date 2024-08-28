import torch
import torch.nn as nn

import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, input_d, action_d, hidden_d=256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_d, hidden_d)
        self.fc2 = nn.Linear(hidden_d, action_d)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

if __name__ == "__main__":
    # 检查 PyTorch 版本
    print(torch.__version__)
    # 检查 GPU 是否可用
    print(torch.cuda.is_available())
    
    state_d = 12
    action_d = 2
    model = DQN(state_d, action_d)
    x = torch.randn(1, state_d)
    print(model)
    print(model(x))
