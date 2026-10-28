# agents/maddpg.py - Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with Cross-Site Decision Making. This file implements the core components of the MADDPG algorithm.

import torch, numpy as np, torch.nn.functional as F
from agents.single_agent import Actor, ReplayBuffer
from rl_charge_extensions import STHGCNet, build_adj

# Encoder MLP
class EncMLP(torch.nn.Module):
    def __init__(self, in_dim, hid=128, out_dim=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, out_dim))
    def forward(self, x):
        return self.net(x)

# Attention-based Critic
class AttnCritic(torch.nn.Module):
    def __init__(self, obs_dim, k, d_future=0, hid=64):
        super().__init__()
        self.k = k
        # input = obs + action + future_obs 
        input_dim = obs_dim + 1 + d_future * 2
        print(f"[DEBUG] AttnCritic init: obs_dim={obs_dim}, k={k}, d_future={d_future}, input_dim={input_dim}")
        
        self.enc = EncMLP(input_dim, hid, hid)
        self.v = torch.nn.Parameter(torch.randn(1, hid))
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, 1))
            
    def forward(self, local_feat):
        # local_feat: (B*N, k, obs+1+d_future*2)
        e = self.enc(local_feat)
        w = torch.softmax(torch.tanh(e) @ self.v.T, dim=1)
        z = (w * e).sum(dim=1)
        return self.out(z)

class CrossSiteActor(torch.nn.Module):
    """Actor Network supporting redirect decisions"""
    def __init__(self, state_dim, hidden_dim=128, n_sites=None, redirect_radius=4):
        super().__init__()
        self.n_sites = n_sites
        self.redirect_radius = redirect_radius
        # main network
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        
        # two heads: acceptance probability + redirect willingness
        self.accept_head = torch.nn.Linear(hidden_dim, 1)      # acceptance probability
        self.redirect_head = torch.nn.Linear(hidden_dim, 1)    # redirect willingness

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))

        # acceptance probability
        accept_prob = torch.sigmoid(self.accept_head(h))

        # redirect willingness
        redirect_willingness = torch.sigmoid(self.redirect_head(h))
        
        return torch.cat([accept_prob, redirect_willingness], dim=-1)

class MADDPG:
    def __init__(self, obs_dim, n_agents, device, env, args, k=5):
        self.N, self.obs_dim, self.dev = n_agents, obs_dim, device
        self.gamma, self.tau = args.gamma, args.tau
        self.batch, self.noise_sd, self.k = args.batch_size, 0.1, k

        # check if cross-site is enabled
        self.enable_cross_site = getattr(env, 'enable_cross_site', False)
        self.redirect_radius = getattr(env, 'redirect_radius', 3)
        print(f"MADDPG initialized with cross_site={self.enable_cross_site}, redirect_radius={self.redirect_radius}")

        # hyperparameters
        self.beta_entropy = getattr(args, 'beta_entropy', 0.01)
        self.lambda_match = getattr(args, 'lambda_match', 0.5)
        self.target_sigma = getattr(args, 'target_sigma', 0.1)

        # initialize
        self.A_hat = build_adj(env.get_knn(), self.N).to(device)
        self.encoder = STHGCNet(self.N, obs_dim,
                                hidden_dim=args.hiddim,
                                out_dim=obs_dim).to(device)

        # if cross-site is enabled, use the CrossSiteActor
        if self.enable_cross_site:
            self.actor = CrossSiteActor(
                state_dim=obs_dim, 
                hidden_dim=args.hiddim,
                n_sites=n_agents,
                redirect_radius=self.redirect_radius
            ).to(device)
            self.actor_t = CrossSiteActor(
                state_dim=obs_dim,
                hidden_dim=args.hiddim,
                n_sites=n_agents,
                redirect_radius=self.redirect_radius
            ).to(device)
        else:
            self.actor = Actor(args, state_dim=obs_dim).to(device)
            self.actor_t = Actor(args, state_dim=obs_dim).to(device)
            
        self.actor_t.load_state_dict(self.actor.state_dict())

        # future feature dimension
        d_f = env.d_future
        self.knn_idx = torch.from_numpy(env.get_knn()).long().to(device)

        # Critic = obs + act(1) + future feature dimension
        critic_obs_dim = obs_dim  
        
        self.critic = AttnCritic(critic_obs_dim, k, d_future=d_f, hid=args.hiddim).to(device)
        self.critic_t = AttnCritic(critic_obs_dim, k, d_future=d_f, hid=args.hiddim).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        # optimizer
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=args.actor_lr)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # diagnostic variables
        self.action_stats = {'redirect_attempts': 0, 'successful_redirects': 0, 'peak_reductions': 0}
        self.reward_components = {'base_reward': [], 'peak_penalty': [], 'redirect_bonus': []}
        
        self.memory = ReplayBuffer(args.buffer_size)
        
    def _build_local(self, feat_s, feat_a, feat_p):
        """modify to support multi-dimensional actions"""
        dev = self.knn_idx.device
        feat_s = feat_s.to(dev)
        feat_a = feat_a.to(dev)
        feat_p = feat_p.to(dev)
        
        B, N, F = feat_s.shape
        k = self.k
        idx = self.knn_idx.unsqueeze(0).expand(B, -1, -1)

        # obtain neighbors' observations
        knn_obs = torch.gather(
            feat_s.unsqueeze(2).expand(B, N, k, F),
            1,
            idx.unsqueeze(-1).expand(B, N, k, F)
        )

        # process actions (which may be multi-dimensional)
        if self.enable_cross_site and feat_a.dim() == 3 and feat_a.shape[2] > 1:
            # only use the accepted probability part
            feat_a_accept = feat_a[:, :, 0:1]
        else:
            feat_a_accept = feat_a
            if feat_a_accept.dim() == 2:
                feat_a_accept = feat_a_accept.unsqueeze(-1)
                
        knn_act = torch.gather(
            feat_a_accept.unsqueeze(2).expand(B, N, k, 1),
            1,
            idx.unsqueeze(-1).expand(B, N, k, 1)
        )

        # obtain neighbors' future features
        knn_p = torch.gather(
            feat_p.unsqueeze(2).expand(B, N, k, feat_p.size(-1)),
            1,
            idx.unsqueeze(-1).expand(B, N, k, feat_p.size(-1))
        )
        
        #print(f"[DEBUG] knn_obs: {knn_obs.shape}, knn_act: {knn_act.shape}, knn_p: {knn_p.shape}")

        local = torch.cat([knn_obs, knn_act, knn_p], dim=-1)
        result = local.view(-1, k, F + 1 + feat_p.size(-1))
        
        # print(f"[DEBUG] _build_local output shape: {result.shape}")
        return result
    
    def update(self):
        """update function to fix device issues"""
        if len(self.memory) < self.batch:
            return 0.0, 0.0

        # sample experience
        s, a_prob, r, s2, d, future_free, future_dom, a_bin = self.memory.sample(self.batch)
        
        s = s.to(self.dev)
        a_prob = a_prob.to(self.dev)
        r = r.to(self.dev)
        s2 = s2.to(self.dev)
        d = d.to(self.dev)
        future_free = future_free.to(self.dev)
        future_dom = future_dom.to(self.dev)
        a_bin = a_bin.to(self.dev)

        # check reward dimensions and adjust
        if r.dim() == 3 and r.shape[2] == 1:
            r = r.squeeze(-1)  # (batch, N)
        elif r.dim() == 1:
            r = r.unsqueeze(0)  # (1, N)

        # check future feature dimensions
        if future_free.dim() == 2:
            future_free = future_free.unsqueeze(-1)
        if future_dom.dim() == 2:
            future_dom = future_dom.unsqueeze(-1)
        future_feat = torch.cat([future_free, future_dom], dim=-1)

        # encode states
        emb_s = self._encode(s)
        emb_s2 = self._encode(s2)

        # construct local features
        local_s_bin = self._build_local(emb_s.detach(), a_bin.detach(), future_feat).to(self.dev)
        
        with torch.no_grad():
            # target network actions
            a2 = self.actor_t(emb_s2)
            if self.target_sigma > 0:
                a2 = torch.clamp(a2 + torch.randn_like(a2) * self.target_sigma, 0., 1.)
            a2_bin = torch.bernoulli(a2)
                
            local_s2 = self._build_local(emb_s2.detach(), a2_bin, future_feat).to(self.dev)
            d_flat = d.expand(-1, self.N).reshape(-1, 1).to(self.dev)

        # ========== Critic update ==========
        # use site-level rewards
        r_site = r.reshape(-1, 1).to(self.dev)  # (batch*N, 1)

        # adjust reward scaling to maintain peak penalty strength
        r_scaled = torch.clamp(r_site, -1000.0, 200.0) / 50.0  

        # compute target values
        with torch.no_grad():
            next_values = self.critic_t(local_s2)
            y = r_scaled + self.gamma * (1 - d_flat) * next_values

        # compute current Q values
        q = self.critic(local_s_bin)

        # TD error
        td_error = (q - y).abs().mean().item()

        # use Huber Loss for stability
        loss_c = F.smooth_l1_loss(q, y)
        
        self.opt_c.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.opt_c.step()

        # ========== Actor update ==========
        pred_a = self.actor(emb_s)

        # extract features from state
        if s.dim() == 3:
            time_feat = s[:, :, 1].to(self.dev)
            queue_feat = s[:, :, 2].to(self.dev)       # queue length
            load_feat = s[:, :, 3].to(self.dev)        # current load
            is_peak_feat = s[:, :, 4].to(self.dev)     # peak flag
            wait_time_feat = s[:, :, 5].to(self.dev) if s.shape[2] > 5 else torch.zeros_like(load_feat)
        else:
            # handle 2D case
            queue_feat = s[:, 2].to(self.dev)
            load_feat = s[:, 3].to(self.dev)
            is_peak_feat = s[:, 4].to(self.dev)
            wait_time_feat = s[:, 5].to(self.dev) if s.shape[1] > 5 else torch.zeros_like(load_feat)

        # extract acceptance probability
        if self.enable_cross_site and pred_a.dim() == 3:
            accept_prob = pred_a[:, :, 0].to(self.dev)
        else:
            accept_prob = pred_a.squeeze(-1).to(self.dev) if pred_a.dim() == 3 else pred_a.to(self.dev)

        # 1. wait time penalty
        wait_penalty = torch.tensor(0.0, device=self.dev)
        high_wait_mask = (wait_time_feat > 0.8).float()  # wait time > 80% max
        low_accept_mask = (accept_prob < 0.3).float()    # low acceptance probability

        # penalty: high wait time but low acceptance probability
        wait_penalty = (high_wait_mask * low_accept_mask * 10.0).mean()

        # 2. peak control loss
        peak_penalty = torch.tensor(0.0, device=self.dev)
        is_peak_mask = (is_peak_feat > 0.5).to(self.dev)

        if is_peak_mask.any():
            # only consider high load sites
            high_load_mask = (load_feat > 0.7).float().to(self.dev)
            peak_violation_mask = (is_peak_mask.float() * high_load_mask).to(self.dev)
            
            if peak_violation_mask.sum() > 0:
                # gentle peak penalty
                peak_penalty = (peak_violation_mask * accept_prob * 5.0).mean()

        # 3. queue backlog loss
        queue_penalty = torch.tensor(0.0, device=self.dev)
        long_queue_mask = (queue_feat > 0.6).float()  # queue > 60%
        low_load_mask = (load_feat < 0.6).float()     # low load

        # penalty: long queue but low load
        queue_penalty = (long_queue_mask * low_load_mask * (1 - accept_prob) * 3.0).mean()

        # 4. balance loss
        # encourage acceptance during medium load
        medium_load_mask = ((load_feat > 0.3) & (load_feat < 0.6)).float()
        medium_queue_mask = (queue_feat > 0.2).float()
        balance_bonus = (medium_load_mask * medium_queue_mask * accept_prob * 2.0).mean()

        # build local features for Q value computation
        pred_a_bin = torch.bernoulli(pred_a)
        local_sa = self._build_local(emb_s, pred_a_bin, future_feat)
        q_sa = self.critic(local_sa)

        # policy gradient loss
        pg_loss = -q_sa.mean()

        # entropy regularization (small weight)
        eps = 1e-8
        if self.enable_cross_site and pred_a.dim() == 3:
            entropy = -(pred_a[:, :, 0] * (pred_a[:, :, 0] + eps).log() + 
                    (1 - pred_a[:, :, 0]) * (1 - pred_a[:, :, 0] + eps).log()).mean()
        else:
            entropy = -(pred_a * (pred_a + eps).log() + 
                    (1 - pred_a) * (1 - pred_a + eps).log()).mean()

        # total loss
        loss_a = (pg_loss - 
                self.beta_entropy * 0.3 * entropy +    # reduce entropy weight
                wait_penalty * 2.0 +                   # increase wait penalty weight
                peak_penalty * 0.5 +                   # reduce peak penalty weight
                queue_penalty * 1.5 -                  # queue backlog penalty
                balance_bonus)                         # balance bonus

        # gradient clipping
        self.opt_enc.zero_grad()
        self.opt_a.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_enc.step()
        self.opt_a.step()

        # restore learning rate
        if td_error > 20.0:
            for param_group in self.opt_a.param_groups:
                param_group['lr'] /= 0.1
            for param_group in self.opt_enc.param_groups:
                param_group['lr'] /= 0.1

        # soft update
        tau = min(self.tau * 2, 0.01)  # slightly accelerate target network update
        with torch.no_grad():
            for tgt, src in zip(self.actor_t.parameters(), self.actor.parameters()):
                tgt.data.copy_(tgt.data * (1 - tau) + src.data * tau)
            for tgt, src in zip(self.critic_t.parameters(), self.critic.parameters()):
                tgt.data.copy_(tgt.data * (1 - tau) + src.data * tau)
        
        return loss_a.item(), loss_c.item()
    
    def _encode(self, obs):
        single = (obs.dim() == 2)
        if single:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.A_hat.device)
        emb = self.encoder(obs, self.A_hat)
        return emb.squeeze(0) if single else emb
    
    @torch.no_grad()
    def act(self, obs):
        """Actor supporting cross-site action selection"""
        emb = self._encode(obs)
        out = self.actor(emb)
        
        if self.enable_cross_site:
            # handle acceptance and redirection probabilities separately
            accept_prob = out[:, 0:1]
            redirect_prob = out[:, 1:]

            # smooth acceptance probability
            accept_prob = torch.sigmoid(accept_prob * 5) * 0.9 + 0.05

            # ensure redirection probabilities sum to 1
            redirect_prob = torch.softmax(redirect_prob * 2, dim=-1)
            
            return torch.cat([accept_prob, redirect_prob], dim=-1)
        else:
            # original single action
            out = torch.sigmoid(out * 5)
            out = out * 0.9 + 0.05
            return out
    
    def add_noise(self, acts):
        """Add noise to cross-site actions"""
        if self.enable_cross_site and acts.dim() == 2:
            # only add noise to acceptance probabilities
            accept_prob = acts[:, 0:1]
            redirect_prob = acts[:, 1:]
            
            noisy_accept = (accept_prob + torch.randn_like(accept_prob) * self.noise_sd).clamp(0., 1.)

            # add small noise to redirection probabilities and re-normalize
            noisy_redirect = redirect_prob + torch.randn_like(redirect_prob) * self.noise_sd * 0.1
            noisy_redirect = torch.softmax(noisy_redirect, dim=-1)
            
            return torch.cat([noisy_accept, noisy_redirect], dim=-1)
        else:
            return (acts + torch.randn_like(acts) * self.noise_sd).clamp(0., 1.)
        
    