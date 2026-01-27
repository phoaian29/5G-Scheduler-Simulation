import numpy as np
from typing import Dict, Tuple


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.data_index = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, p: float):
        idx = self.data_index + self.capacity
        self.update(idx, p)
        self.data_index = (self.data_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, idx: int, p: float):
        delta = p - self.tree[idx]
        self.tree[idx] = p
        idx //= 2
        while idx >= 1:
            self.tree[idx] += delta
            idx //= 2

    def get(self, s: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx  # leaf index


class PrioritizedReplayBuffer:
    """
    Stores transitions:
      (state, action, reward, next_state, mask, next_mask)
    """
    def __init__(self, capacity: int, per_alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(per_alpha)
        self.tree = SumTree(self.capacity)

        self.states = [None] * self.capacity
        self.actions = [None] * self.capacity
        self.rewards = [None] * self.capacity
        self.next_states = [None] * self.capacity
        self.masks = [None] * self.capacity
        self.next_masks = [None] * self.capacity

        self.max_p = 1.0

    def __len__(self):
        return self.tree.size

    def add(self, state, action, reward, next_state, mask, next_mask):
        p = (self.max_p + 1e-6) ** self.alpha
        leaf_idx = self.tree.add(p)
        data_idx = leaf_idx - self.capacity

        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_states[data_idx] = next_state
        self.masks[data_idx] = mask
        self.next_masks[data_idx] = next_mask

    def sample(self, batch_size: int, beta: float) -> Dict[str, np.ndarray]:
        batch_size = int(batch_size)
        beta = float(beta)

        total = self.tree.total()
        seg = total / batch_size

        indices = np.empty(batch_size, dtype=np.int32)
        weights = np.empty(batch_size, dtype=np.float32)

        states, actions, rewards, next_states, masks, next_masks = [], [], [], [], [], []

        # compute min prob for weight normalization
        leaf_start = self.capacity
        ps = self.tree.tree[leaf_start:leaf_start + self.tree.size]
        min_p = np.min(ps) / total if self.tree.size > 0 else 1.0
        max_w = (self.tree.size * min_p) ** (-beta) if min_p > 0 else 1.0

        for i in range(batch_size):
            s = np.random.uniform(seg * i, seg * (i + 1))
            leaf = self.tree.get(s)
            data_idx = leaf - self.capacity

            p = self.tree.tree[leaf] / total
            w = (self.tree.size * p) ** (-beta)
            weights[i] = w / (max_w + 1e-12)
            indices[i] = leaf

            states.append(self.states[data_idx])
            actions.append(self.actions[data_idx])
            rewards.append(self.rewards[data_idx])
            next_states.append(self.next_states[data_idx])
            masks.append(self.masks[data_idx])
            next_masks.append(self.next_masks[data_idx])

        return {
            "state": np.stack(states),
            "action": np.stack(actions),
            "reward": np.stack(rewards),
            "next_state": np.stack(next_states),
            "mask": np.stack(masks),
            "next_mask": np.stack(next_masks),
            "weights": weights,
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        priorities = np.asarray(priorities, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32)
        self.max_p = max(self.max_p, float(np.max(priorities)))

        for leaf, pr in zip(indices, priorities):
            p = (float(pr) + 1e-6) ** self.alpha
            self.tree.update(int(leaf), p)
