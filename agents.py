import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, n_heads=1, hidden=128):
        super().__init__()
        self.n_heads = n_heads
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n_actions) for _ in range(n_heads)])

    def forward(self, x):
        feat = self.shared(x)
        return torch.stack([head(feat) for head in self.heads], dim=1)  # (batch, n_heads, n_actions)


class DQNAgentBase:
    def __init__(self, state_dim, n_actions, n_heads=1, channel_weights=None,
                 lr=1e-3, gamma=0.99, buffer_size=100_000, batch_size=64,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=20_000, device="cpu"):
        self.n_actions = n_actions
        self.n_heads = n_heads
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)

        weights = channel_weights if channel_weights is not None else [1.0] * n_heads
        assert len(weights) == n_heads, "channel_weights 길이는 n_heads와 같아야 함"
        self.channel_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        self.q_net = QNetwork(state_dim, n_actions, n_heads).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions, n_heads).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.step_count = 0

    def epsilon(self):
        frac = min(1.0, self.step_count / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _combine(self, q):
        # q: (batch, n_heads, n_actions) -> (batch, n_actions)
        return (q * self.channel_weights.view(1, -1, 1)).sum(dim=1)

    def act(self, state):
        self.step_count += 1
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_combined = self._combine(self.q_net(s))
            return int(q_combined.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        dones = torch.tensor(dones, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)  # (batch,1) -> 모든 head에 동일 보상(대조군)

        q_values = self.q_net(states)  # (batch, n_heads, n_actions)
        idx = actions.view(-1, 1, 1).expand(-1, self.n_heads, 1)
        q_taken = q_values.gather(2, idx).squeeze(2)  # (batch, n_heads)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_actions = self._combine(next_q).argmax(dim=1)
            next_idx = next_actions.view(-1, 1, 1).expand(-1, self.n_heads, 1)
            next_q_taken = next_q.gather(2, next_idx).squeeze(2)
            target = rewards + self.gamma * next_q_taken * (1 - dones)

        loss_per_elem = F.smooth_l1_loss(q_taken, target, reduction="none")  # (batch, n_heads)
        loss = loss_per_elem.mean()  # 학습(backward)에는 항상 이 전체 평균을 사용하도록 함.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        per_channel_loss = loss_per_elem.mean(dim=0).detach().cpu().tolist()  # 로깅 전용
        return loss.item(), per_channel_loss

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


def make_baseline_agent(state_dim, n_actions, **kwargs):
    return DQNAgentBase(state_dim, n_actions, n_heads=1, channel_weights=[1.0], **kwargs)


def make_fly_hra_agent(state_dim, n_actions, channel_weights=(1.0, 1.0, 1.0), **kwargs):
    return DQNAgentBase(state_dim, n_actions, n_heads=3, channel_weights=list(channel_weights), **kwargs)
