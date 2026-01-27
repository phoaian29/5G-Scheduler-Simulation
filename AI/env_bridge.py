import numpy as np


class MatlabEnvBridge:
    def __init__(self, max_ues: int, rho: float = 0.9):
        self.max_ues = int(max_ues)
        self.rho = float(rho)

        self.avg_tput = np.zeros(self.max_ues, dtype=np.float32)
        self.prev_pf = None

        # for delayed transition
        self.prev_state = None
        self.prev_mask = None
        self.prev_action = None

    def build_state_and_mask(self, payload: dict):
        feats = np.asarray(payload["features"], dtype=np.float32)  # [U, 5+2M]
        num_subbands = int(payload["num_subbands"])
        feat_dim = feats.shape[1]
        assert feat_dim == 5 + 2 * num_subbands, "features dim mismatch"

        # flatten state for networks: [state_dim]
        state = feats.reshape(-1)  # U*(5+2M)

        # action space: A = max_ues + 1 (last = no-allocation)
        A = self.max_ues + 1
        M = num_subbands

        mask = np.zeros((M, A), dtype=bool)

        # eligible UE detection: row not all zeros
        row_nonzero = np.any(np.abs(feats) > 0, axis=1)  # [U]

        # buffer feature is index 3 in first 5
        buffer_ok = feats[:, 3] > 0
        ue_ok = row_nonzero & buffer_ok

        # subband CQI vector starts at index 5, length M
        sb_cqi = feats[:, 5:5 + M]  # [U,M]

        for m in range(M):
            valid_ues_m = ue_ok & (sb_cqi[:, m] > 0)
            mask[m, :self.max_ues] = valid_ues_m
            mask[m, self.max_ues] = True  # no-allocation always valid

        return state.astype(np.float32), mask

    def action_to_prbs(self, actions: np.ndarray, num_subbands: int, prb_budget: int, subband_size: int):
        """
        actions: [M] values 0..max_ues (max_ues is no-allocation)
        returns prbs: [max_ues]
        """
        prbs = np.zeros(self.max_ues, dtype=np.int32)
        M = int(num_subbands)

        # each subband corresponds to subband_size PRBs except last maybe truncated by prb_budget
        for m in range(M):
            ue = int(actions[m])
            start = m * subband_size
            end = min((m + 1) * subband_size, prb_budget)
            alloc = max(0, end - start)
            if alloc <= 0:
                continue
            if ue < self.max_ues:
                prbs[ue] += alloc
        return prbs

    def compute_reward_from_last_served(self, last_served_bytes: np.ndarray):
        """
        last_served_bytes: [max_ues]
        Use it as instantaneous throughput proxy to update avg_tput and compute PF gain.
        """
        last_served_bytes = np.asarray(last_served_bytes, dtype=np.float32)
        inst = last_served_bytes  # proxy; you can scale to Mbps if you want

        self.avg_tput = self.rho * self.avg_tput + (1.0 - self.rho) * inst

        eps = 1.0  # avoid log(0)
        pf = float(np.exp(np.mean(np.log(self.avg_tput + eps))))

        if self.prev_pf is None:
            self.prev_pf = pf
            return 0.0

        delta = pf - self.prev_pf
        self.prev_pf = pf

        # normalize to [-1,1]
        reward = float(np.tanh(delta / (pf + 1e-6)))
        return reward
