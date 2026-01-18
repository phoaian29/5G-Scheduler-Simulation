import socket
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rl_mac_env import project_scores_to_prbs, bytes_per_prb, jain_fairness

class MatlabBridgeEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=5555, prb_budget=51):
        super().__init__()
        
        # Configuration: 5 Features * 4 UEs = 20 input dimensions
        self.obs_dim = 20 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self.host = host
        self.port = port
        self.prb_budget = prb_budget
        
        self.sock = None
        self.conn = None
        
        self.last_raw_state = None
        self.current_step = 0 
        self.matlab_finished = False 
        
        self.thr_ema_mbps = np.ones(4, dtype=float) * 1e-6
        self.rho = 0.9 
        self.max_mb_per_tti = (self.prb_budget * bytes_per_prb(28) * 8.0) / 1e6
        # Receive timeout handling - increased for CSI RS processing
        self._recv_timeouts = 0
        self._recv_timeout_limit = 20  # Increased from 8 to 20 retries
        
        self.setup_server()

    def setup_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(300)  # 5-minute timeout for acceptance
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"[TRAINER] Waiting for MATLAB connection at {self.host}:{self.port}...")
        self.conn, addr = self.sock.accept()
        # Increased timeout for CSI RS processing which takes longer
        self.conn.settimeout(30)  # 30-second timeout for each receive attempt
        print(f"[TRAINER] MATLAB connected from {addr}")
        self.matlab_finished = False

    def reset(self, seed=None, options=None):
        if self.matlab_finished:
            print("[INFO] MATLAB simulation already finished. Cannot reset.")
            print("[STOP] Training should stop now via StopOnMatlabFinishedCallback.")
            return np.zeros(self.obs_dim, dtype=np.float32), {}

        # Just wait for initial state from MATLAB (no need to request)
        print("[TRAINER] Waiting for initial state from MATLAB...")
        raw_state = self._recv_from_matlab()
        
        if raw_state is None:
            print("[ERROR] MATLAB connection closed immediately")
            self.matlab_finished = True
            return np.zeros(self.obs_dim, dtype=np.float32), {}

        if 'prb_budget' in raw_state: 
            self.prb_budget = int(raw_state['prb_budget'])
            self.max_mb_per_tti = (self.prb_budget * bytes_per_prb(28) * 8.0) / 1e6
        
        self.last_raw_state = raw_state
        obs = self._compute_obs(raw_state)
        self.thr_ema_mbps = np.ones(4, dtype=float) * 1e-6
        self.current_step = 0
        print("[TRAINER] Received initial state, starting training...")
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        prev_feats = np.array(self.last_raw_state['features']) 
        backlog_vec = prev_feats[:, 0]
        cqi_vec = prev_feats[:, 2]
        
        mcs_vec = np.clip(cqi_vec * 1.8, 0, 28).astype(int) 
        active_mask = (backlog_vec > 0).astype(int)
        scores = np.clip(action, 0.0, 1.0)
        
        try:
            prbs_out, _, wasted_prbs, _ = project_scores_to_prbs(
                scores, self.prb_budget, backlog_vec, mcs_vec, active_mask, training_mode=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to project scores to PRBs: {e}")
            print(f"  scores: {scores}, budget: {self.prb_budget}, backlog: {backlog_vec}, mcs: {mcs_vec}, mask: {active_mask}")
            raise
        
        if not self._send_to_matlab({"prbs": prbs_out.tolist()}):
            print("[ERROR] Failed to send data to MATLAB. Connection closed.")
            self.matlab_finished = True
            return self._compute_obs(self.last_raw_state), 0.0, True, False, {
                "cell_tput_Mb": 0.0,
                "jain": 0.0,
                "wasted_prbs": 0
            }

        next_raw_state = self._recv_from_matlab()
        
        if next_raw_state is None:
            print(f"[INFO] MATLAB simulation finished at step {self.current_step}")
            self.matlab_finished = True
            return self._compute_obs(self.last_raw_state), 0.0, True, False, {
                "cell_tput_Mb": 0.0,
                "jain": 0.0,
                "wasted_prbs": 0
            }

        if 'prb_budget' in next_raw_state: self.prb_budget = int(next_raw_state['prb_budget'])
        self.last_raw_state = next_raw_state
        
        served_bytes = np.array(next_raw_state['last_served'])
        served_mb_tti = (served_bytes.sum() * 8.0) / 1e6
        tput_norm = served_mb_tti / (self.max_mb_per_tti + 1e-12)
        inst_rate_mbps = (served_bytes * 8.0) / 1e6 / 0.001 
        self.thr_ema_mbps = self.rho * self.thr_ema_mbps + (1 - self.rho) * inst_rate_mbps
        jain = jain_fairness(self.thr_ema_mbps)
        
        reward = 1.0 * tput_norm + 0.2 * jain
        next_obs = self._compute_obs(next_raw_state)
        
        info = {
            "cell_tput_Mb": float(served_mb_tti),
            "jain": float(jain),
            "wasted_prbs": int(wasted_prbs)
        }
        
        if self.current_step % 10 == 0:
            print("-" * 60)
            print(f"Step {self.current_step:04d} | Total Reward: {reward:.4f}")
            print(f"[Input] PRB Budget: {self.prb_budget}")
            print(f"   UE Buffer (Bytes): {backlog_vec.astype(int)}")
            print(f"[Agent] Action Scores: {scores.round(2)}")
            print(f"[Output] PRB Allocation: {prbs_out} | Wasted: {wasted_prbs}")
            print("-" * 60)

        return next_obs, float(reward), False, False, info

    def _compute_obs(self, raw):
        feats = np.array(raw['features']) 
        obs_vec = []
        MAX_BYTES = self.prb_budget * 1000.0 
        for i in range(4):
            obs_vec.append(np.clip(feats[i][0] / MAX_BYTES, 0, 1))  # Buffer size
            obs_vec.append(np.clip(feats[i][1] / 1000.0, 0, 1))     # Throughput
            obs_vec.append(np.clip(feats[i][2] / 15.0, 0, 1))       # CQI
            obs_vec.append(np.clip(feats[i][3] / 4.0, 0, 1))        # Rank
            obs_vec.append(np.clip(feats[i][4], 0, 1))              # Allocation ratio
        return np.array(obs_vec, dtype=np.float32)

    def _recv_from_matlab(self):
        buffer = ""
        while "\n" not in buffer:
            try:
                data = self.conn.recv(4096)
                if not data: return None
                chunk = data.decode('utf-8')
                buffer += chunk
                # Debug: raw chunk received
                print(f"[MATLAB->TRAINER RAW] {chunk.strip()}")
                # Reset timeout counter after receiving any data
                self._recv_timeouts = 0
            except socket.timeout:
                # Transient timeout: allow several retries before concluding MATLAB finished
                self._recv_timeouts += 1
                print(f"[TRAINER] recv timeout #{self._recv_timeouts}/{self._recv_timeout_limit}")
                if self._recv_timeouts >= self._recv_timeout_limit:
                    print("[TRAINER] recv timeout limit reached — treating MATLAB as finished")
                    return None
                else:
                    continue
            except Exception:
                print("[TRAINER] unexpected exception during recv")
                return None
        line, _ = buffer.split("\n", 1)
        try:
            # Debug: full JSON line from MATLAB
            print(f"[MATLAB->TRAINER LINE] {line}")
            # Reset timeout counter on successful parse
            self._recv_timeouts = 0
            return json.loads(line)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON from MATLAB: {e} | line={line}")
            return None

    def _send_to_matlab(self, data):
        try:
            payload = json.dumps(data)
            # Debug: payload being sent to MATLAB
            print(f"[TRAINER->MATLAB] {payload}")
            self.conn.sendall((payload + "\n").encode('utf-8'))
            return True
        except socket.timeout:
            return False
        except Exception:
            return False

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'sock' in state: del state['sock']
        if 'conn' in state: del state['conn']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.sock = None
        self.conn = None