import copy
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSACDUpdateInfo:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    mean_priority: float


class DSACDAgent:
    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        replay_buffer,
        *,
        device: torch.device,
        n_quantiles: int = 16,
        gamma: float = 0.0,
        tau: float = 0.005,
        beta_entropy: float = 0.98,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        grad_clip_norm: Optional[float] = 10.0,
        huber_kappa: float = 1.0,
        per_eps: float = 1e-6,
    ):
        self.device = device
        self.actor = actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)

        self.target_critic1 = copy.deepcopy(self.critic1).to(device).eval()
        self.target_critic2 = copy.deepcopy(self.critic2).to(device).eval()
        for p in self.target_critic1.parameters():
            p.requires_grad_(False)
        for p in self.target_critic2.parameters():
            p.requires_grad_(False)

        self.replay_buffer = replay_buffer

        self.N = int(n_quantiles)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.beta = float(beta_entropy)
        self.grad_clip_norm = grad_clip_norm
        self.kappa = float(huber_kappa)
        self.per_eps = float(per_eps)

        self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        taus = (torch.arange(1, self.N + 1, device=device, dtype=torch.float32) / self.N)
        self.taus = taus.view(1, 1, 1, self.N)  # [1,1,1,N]

        self._updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        state: [state_dim] or [B,state_dim]
        action_mask: bool [NRBG, A] or [B,NRBG,A] (True=valid)
        returns: actions [NRBG] or [B,NRBG]
        """
        state = state.to(self.device)
        mask = action_mask.to(self.device).bool()

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        logits = self.actor(state)  # [B,NRBG,A]
        logits = self._mask_logits(logits, mask)

        if deterministic:
            a = logits.argmax(dim=-1)
            return a.squeeze(0)

        probs = F.softmax(logits, dim=-1)
        probs = self._renormalize_probs(probs, mask)

        B, NRBG, _ = probs.shape
        a = torch.empty((B, NRBG), device=self.device, dtype=torch.long)
        for m in range(NRBG):
            dist = torch.distributions.Categorical(probs=probs[:, m, :])
            a[:, m] = dist.sample()
        return a.squeeze(0)

    def update_parameters(self, batch_size: int, per_beta: float) -> DSACDUpdateInfo:
        batch_np = self.replay_buffer.sample(batch_size, beta=per_beta)

        s = torch.tensor(batch_np["state"], device=self.device, dtype=torch.float32)
        a = torch.tensor(batch_np["action"], device=self.device, dtype=torch.long)
        r = torch.tensor(batch_np["reward"], device=self.device, dtype=torch.float32)
        s2 = torch.tensor(batch_np["next_state"], device=self.device, dtype=torch.float32)

        mask = torch.tensor(batch_np["mask"], device=self.device, dtype=torch.bool)
        mask2 = torch.tensor(batch_np["next_mask"], device=self.device, dtype=torch.bool)

        w_is = torch.tensor(batch_np["weights"], device=self.device, dtype=torch.float32)
        indices = batch_np["indices"]

        # reward: [B] -> broadcast to [B,NRBG] if needed
        if r.dim() == 1 and a.dim() == 2:
            r = r.unsqueeze(-1).expand_as(a)

        # ---- Critic target ----
        with torch.no_grad():
            next_logits = self.actor(s2)
            next_logits = self._mask_logits(next_logits, mask2)
            next_probs = F.softmax(next_logits, dim=-1)
            next_probs = self._renormalize_probs(next_probs, mask2)

            a2 = self._sample_actions(next_probs)                  # [B,NRBG]
            logp_a2 = self._logp(next_probs, a2)                   # [B,NRBG]

            tq1_all = self.target_critic1(s2)                      # [B,NRBG,A,N]
            tq2_all = self.target_critic2(s2)

            tq1_a2 = self._gather_quantiles(tq1_all, a2)           # [B,NRBG,N]
            tq2_a2 = self._gather_quantiles(tq2_all, a2)           # [B,NRBG,N]

            tq_min_mean = torch.min(tq1_a2.mean(-1), tq2_a2.mean(-1))  # [B,NRBG]
            y = r + self.gamma * (tq_min_mean - self.alpha.detach() * logp_a2)  # [B,NRBG]
            y = y.unsqueeze(-1).expand(-1, -1, self.N)              # [B,NRBG,N]

        cq1_all = self.critic1(s)
        cq2_all = self.critic2(s)
        cq1 = self._gather_quantiles(cq1_all, a)
        cq2 = self._gather_quantiles(cq2_all, a)

        td1 = cq1 - y
        td2 = cq2 - y

        qh1 = self._quantile_huber(td1)
        qh2 = self._quantile_huber(td2)

        critic1_loss = (qh1.mean(dim=(1, 2)) * w_is).mean()
        critic2_loss = (qh2.mean(dim=(1, 2)) * w_is).mean()
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optim.zero_grad(set_to_none=True)
        self.critic2_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_norm)
        self.critic1_optim.step()
        self.critic2_optim.step()

        # PER priorities: avg abs TD over (2 critics × N × NRBG)
        with torch.no_grad():
            abs_td = 0.5 * (td1.abs() + td2.abs())
            prios = abs_td.mean(dim=(1, 2)) + self.per_eps  # [B]
            mean_prio = float(prios.mean().item())
            self.replay_buffer.update_priorities(indices, prios.detach().cpu().numpy())

        # ---- Actor ----
        logits = self.actor(s)
        logits = self._mask_logits(logits, mask)
        probs = F.softmax(logits, dim=-1)
        probs = self._renormalize_probs(probs, mask)
        log_probs = torch.log(torch.clamp(probs, min=1e-12))

        with torch.no_grad():
            q1_all = self.critic1(s).mean(dim=-1)  # [B,NRBG,A]
            q2_all = self.critic2(s).mean(dim=-1)
            q_min = torch.min(q1_all, q2_all)
            q_min = q_min.masked_fill(~mask, 0.0)

        actor_loss = (probs * (self.alpha.detach() * log_probs - q_min)).sum(dim=-1).mean()

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optim.step()

        # ---- Alpha ---- (state-specific target entropy using number of valid actions)
        with torch.no_grad():
            valid_counts = mask.sum(dim=-1).clamp(min=1).float()  # [B,NRBG]
            target_entropy = -self.beta * torch.log(1.0 / valid_counts)  # [B,NRBG]

        alpha_loss = (
            self.alpha * (probs.detach() * (log_probs.detach() + target_entropy.unsqueeze(-1))).sum(dim=-1)
        ).mean()

        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        # ---- Target soft update ----
        self._soft_update(self.target_critic1, self.critic1, self.tau)
        self._soft_update(self.target_critic2, self.critic2, self.tau)

        return DSACDUpdateInfo(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.item()),
            mean_priority=mean_prio,
        )

    # ---------------- helpers ----------------
    def _mask_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.finfo(logits.dtype).min
        return logits.masked_fill(~mask, neg_inf)

    def _renormalize_probs(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = probs * mask.float()
        z = probs.sum(dim=-1, keepdim=True)
        safe = z > 0
        probs = torch.where(safe, probs / z.clamp(min=1e-12), probs)
        if not torch.all(safe):
            valid = mask.float()
            vcnt = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = valid / vcnt
            probs = torch.where(safe, probs, uniform)
        return probs

    def _sample_actions(self, probs: torch.Tensor) -> torch.Tensor:
        B, NRBG, _ = probs.shape
        a = torch.empty((B, NRBG), device=self.device, dtype=torch.long)
        for m in range(NRBG):
            dist = torch.distributions.Categorical(probs=probs[:, m, :])
            a[:, m] = dist.sample()
        return a

    def _logp(self, probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        p = probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        return torch.log(torch.clamp(p, min=1e-12))

    def _gather_quantiles(self, q_all: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, NRBG, A, N = q_all.shape
        idx = actions.unsqueeze(-1).unsqueeze(-1).expand(B, NRBG, 1, N)
        return q_all.gather(dim=2, index=idx).squeeze(2)

    def _quantile_huber(self, td: torch.Tensor) -> torch.Tensor:
        abs_td = td.abs()
        huber = torch.where(
            abs_td <= self.kappa,
            0.5 * td.pow(2),
            self.kappa * (abs_td - 0.5 * self.kappa),
        )
        indicator = (td < 0).float()
        weight = (self.taus.squeeze(2) - indicator).abs()
        return weight * huber

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
