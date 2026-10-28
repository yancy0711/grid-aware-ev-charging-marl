# rl_charge_extensions.py
# ===============================================================
# Auxiliary components for Reinforcement Learning:
#   1) DelayedReplayBuffer —— Experience replay buffer with "delayed write"
#   2) STHGCNet            —— Minimal spatial-temporal graph convolution encoder
#   3) build_adj           —— knn-idx → normalized adjacency
# ===============================================================
import torch
import numpy as np
from collections import deque

# ---------------------------------------------------------------
# 1. DelayedReplayBuffer
# ---------------------------------------------------------------
class DelayedReplayBuffer:
    """
    When pushing, first place it in _queue, and wait until delay_steps has accumulated before writing it to the main buffer.
    This way, a transition will be sampled at least delay_steps later.
    Other APIs are consistent with ordinary ReplayBuffer.
    """
    def __init__(self, capacity: int, delay_steps: int = 4):
        self._main   = deque(maxlen=capacity)
        self._queue  = deque(maxlen=delay_steps)
        self._delay  = delay_steps

    def push(self, *trans):
        self._queue.append(trans)
        if len(self._queue) == self._delay:
            self._main.append(self._queue.popleft())

    def __len__(self):
        return len(self._main)

    def sample(self, batch_size: int):
        idx  = np.random.choice(len(self._main), batch_size, replace=False)
        cols = list(zip(*[self._main[i] for i in idx]))

        def _to_tensor(x):
            if torch.is_tensor(x):
                return x.float()
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=torch.float32)
            return torch.tensor(x, dtype=torch.float32)

        return [torch.stack([_to_tensor(e) for e in col]) for col in cols]


# ---------------------------------------------------------------
# 2.  STHGCNet (Spatial-Temporal-Hetero Graph Conv, minimal implementation: a minimal spatio-temporal graph convolutional network (GCN) with input (B, N, F), performing two layers of graph convolution using A_hat, and outputting embeddings for each site)
# ---------------------------------------------------------------
class _GCN(torch.nn.Module):
    """ Single layer of A·X·W with ReLU activation """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.w = torch.nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x, A_hat):                   # x (B,N,C)
        return torch.relu(self.w(A_hat @ x))       # (B,N,out_ch)


class STHGCNet(torch.nn.Module):
    """
    Input  obs :
        • (B,N,F)        —— Single time step
        • (B,T,N,F)      —— Sequence (automatically average over T)
    Output emb : (B,N,out_dim)
    """
    def __init__(self,
                 num_nodes : int,
                 in_dim    : int,
                 hidden_dim: int = 64,
                 out_dim   : int = 64):
        super().__init__()
        self.n_nodes = num_nodes
        # self.gcn1    = _GCN(in_dim,    hidden_dim)
        # self.gcn2    = _GCN(hidden_dim, out_dim)
        self.gcn1    = _GCN(in_dim,    out_dim)

    def forward(self, obs, A_hat):
        """
        obs   : (B,N,F) or (B,T,N,F)
        A_hat : (N,N) —— normalized adjacency
        """
        if obs.dim() == 3:                       # (B,N,F)
            x = obs
        elif obs.dim() == 4:                     # (B,T,N,F)
            x = obs.mean(dim=1)                  # average → (B,N,F)
        else:
            raise ValueError(f"Unexpected obs shape {obs.shape}")

        B, N, _ = x.shape
        assert N == self.n_nodes, \
            f"node-num mismatch: encoder expects {self.n_nodes}, got {N}"

        x = self.gcn1(x, A_hat)                  # (B,N,H)
        # x = self.gcn2(x, A_hat)                  # (B,N,out_dim)
        return x


# ---------------------------------------------------------------
# 3. KNN index → normalized adjacency A_hat
# ---------------------------------------------------------------
def build_adj(knn_idx: np.ndarray, N: int):
    """
    knn_idx : (N,K)  —— Each row is an index of the K nearest neighbors of node i.
    return     : A_hat (torch.FloatTensor, N×N)
              is normalized by D^{-1/2} Â D^{-1/2}
    """
    A = torch.zeros((N, N), dtype=torch.float32)
    for i, neigh in enumerate(knn_idx):
        A[i, neigh] = 1
        A[neigh, i] = 1          # Undirected graph: symmetric
    A.fill_diagonal_(1)

    deg   = A.sum(dim=1, keepdim=True)           # (N,1)
    A_hat = A / torch.sqrt(deg @ deg.T + 1e-6)
    return A_hat
