import json
import socket
import time
import numpy as np
import torch

from config import DRLConfig
from networks import Actor1LDS, QuantileCritic1LDS
from per_buffer import PrioritizedReplayBuffer
from dsacd_agent import DSACDAgent
from env_bridge import MatlabEnvBridge


def recv_line(conn: socket.socket) -> bytes:
    buf = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf.extend(chunk)
        if b"\n" in chunk:
            break
    line, _, _ = buf.partition(b"\n")
    return bytes(line)


def main():
    cfg = DRLConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[PY] Starting DRL server on {cfg.host}:{cfg.port} ({device})")

    # We don't know num_subbands until first payload → lazy init networks
    agent = None
    bridge = MatlabEnvBridge(max_ues=cfg.max_ues, rho=0.9)
    replay = PrioritizedReplayBuffer(cfg.replay_size, per_alpha=cfg.per_alpha)

    step = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((cfg.host, cfg.port))
        s.listen(1)
        conn, addr = s.accept()

        with conn:
            print(f"[PY] Connected from {addr}")

            while True:
                line = recv_line(conn)
                payload = json.loads(line.decode("utf-8"))

                # Build current state/mask
                state, mask = bridge.build_state_and_mask(payload)
                num_subbands = int(payload["num_subbands"])
                prb_budget = int(payload["prb_budget"])
                last_served = np.asarray(payload["last_served"], dtype=np.float32)

                # lazy init networks once we know num_subbands
                if agent is None:
                    state_dim = cfg.max_ues * (5 + 2 * num_subbands)
                    n_actions = cfg.max_ues + 1

                    actor = Actor1LDS(state_dim, num_subbands, n_actions, hidden=256)
                    critic1 = QuantileCritic1LDS(state_dim, num_subbands, n_actions, cfg.n_quantiles, hidden=256)
                    critic2 = QuantileCritic1LDS(state_dim, num_subbands, n_actions, cfg.n_quantiles, hidden=256)

                    agent = DSACDAgent(
                        actor, critic1, critic2, replay,
                        device=device,
                        n_quantiles=cfg.n_quantiles,
                        gamma=cfg.gamma,
                        tau=cfg.tau,
                        beta_entropy=cfg.beta_entropy,
                        lr_actor=cfg.lr_actor,
                        lr_critic=cfg.lr_critic,
                        lr_alpha=cfg.lr_alpha,
                    )
                    print(f"[PY] Networks initialized: state_dim={state_dim}, NRBG={num_subbands}, A={n_actions}")

                # ------ delayed reward: when we receive payload(t), last_served is outcome of action(t-1)
                reward = bridge.compute_reward_from_last_served(last_served)

                # store transition if we have previous action
                if bridge.prev_state is not None:
                    replay.add(
                        bridge.prev_state,
                        bridge.prev_action,
                        reward,
                        state,
                        bridge.prev_mask,
                        mask,
                    )

                # select action for current state
                st_t = torch.tensor(state, dtype=torch.float32, device=device)
                mk_t = torch.tensor(mask, dtype=torch.bool, device=device)
                actions = agent.select_action(st_t, mk_t, deterministic=False).detach().cpu().numpy()  # [M]

                # convert actions -> PRB counts per UE for MATLAB
                prbs = bridge.action_to_prbs(
                    actions=actions,
                    num_subbands=num_subbands,
                    prb_budget=prb_budget,
                    subband_size=cfg.subband_size_prb,
                )

                # send response
                resp = {"prbs": prbs.tolist()}
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))

                # remember for next transition
                bridge.prev_state = state
                bridge.prev_mask = mask
                bridge.prev_action = actions.astype(np.int64)

                # training
                step += 1
                if len(replay) >= cfg.warmup_steps and (step % cfg.update_every == 0):
                    beta = cfg.per_beta_start + (cfg.per_beta_end - cfg.per_beta_start) * min(
                        1.0, step / max(1, cfg.per_beta_anneal_steps)
                    )
                    for _ in range(cfg.updates_per_step):
                        info = agent.update_parameters(cfg.batch_size, per_beta=beta)

                    if step % 1 == 0:
                        print(f"[PY] step={step} | alpha={info.alpha:.4f} "
                              f"| Lc={info.critic_loss:.4f} La={info.actor_loss:.4f} "
                              f"| prio={info.mean_priority:.6f}")

if __name__ == "__main__":
    main()
