import torch, torch.nn as nn, numpy as np, random, collections
from collections import deque
# --------------------------------------------------
# ReplayBuffer, used for storing every (state, action, reward, next_state, done) transition into the experience replay buffer.
# --------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)
    def push(self, *transition):
        def to_float32(x):
            if torch.is_tensor(x):
                return x.float()
            elif isinstance(x, np.ndarray):
                return x.astype(np.float32)
            return x
        self.buf.append(tuple(to_float32(t) for t in transition))
    def sample(self, batch_size):
        idx = random.sample(range(len(self.buf)), batch_size)
        trans = list(zip(*[self.buf[i] for i in idx]))
        return [torch.stack(t) if torch.is_tensor(t[0]) else np.stack(t) for t in trans]
    def __len__(self): return len(self.buf)

# --------------------------------------------------
#  Actor Network, 1-D version, used for single-agent scenarios
# --------------------------------------------------
class Actor(nn.Module):   # Action: input state, output action (between 0 and 1, Sigmoid)
    def __init__(self, args, state_dim=None):
        super().__init__()
        hid = getattr(args, 'hiddim', 64) if hasattr(args, 'hiddim') else getattr(args, 'hid', 64) if hasattr(args, 'hid') else 64
        state_dim = state_dim if state_dim is not None else getattr(args, 'state_dim', 3)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1), nn.Sigmoid())
    def forward(self, s): return self.net(s)
