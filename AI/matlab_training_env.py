import socket
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MatlabBridgeEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=5555, max_ues=64, n_rbg=18):
        super().__init__()
        
        self.host = host
        self.port = port
        self.max_ues = max_ues
        self.n_rbg = n_rbg  # Số lượng RBG (ví dụ: 18 cho 100MHz với RBG size lớn, hoặc cấu hình khác)
        
        # --- 1. STATE SPACE (Theo bài báo: Appendix B.2) ---
        # Shape: (MaxUEs, 5 + 2*N_RBG)
        # 5 scalar features + 2 vector features (Subband CQI & Correlation) mỗi UE
        self.feature_dim = 5 + 2 * self.n_rbg
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.max_ues, self.feature_dim), 
            dtype=np.float32
        )
        
        # --- 2. ACTION SPACE (Theo bài báo: Appendix B.1 - 1LDS) ---
        # Output: Logits cho mỗi RBG để chọn UE.
        # Shape: (N_RBG, MaxUEs + 1). "+1" là cho hành động "No Allocation".
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_rbg, self.max_ues + 1),
            dtype=np.float32
        )
        
        self.sock = None
        self.conn = None
        self.last_raw_state = None
        self.setup_server()

    def setup_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"[TRAINER] Waiting for MATLAB at {self.host}:{self.port}...")
        self.conn, addr = self.sock.accept()
        print(f"[TRAINER] MATLAB connected: {addr}")

    def reset(self, seed=None, options=None):
        print("[TRAINER] Waiting for initial state...")
        raw_state = self._recv_from_matlab()
        if raw_state is None:
            return np.zeros((self.max_ues, self.feature_dim), dtype=np.float32), {}
        
        # Cập nhật số lượng RBG thực tế từ MATLAB nếu có
        if 'num_subbands' in raw_state:
            self.n_rbg = int(raw_state['num_subbands'])
            # Lưu ý: Nếu n_rbg thay đổi động, gym space sẽ không khớp. 
            # Tốt nhất là cố định n_rbg max (padding) hoặc cấu hình chuẩn ngay từ đầu.

        self.last_raw_state = raw_state
        obs = self._process_obs(raw_state)
        return obs, {}

    def step(self, action):
        # action shape: (N_RBG, MaxUEs + 1)
        # Bài báo 1LDS: Chọn UE có logit lớn nhất tại mỗi RBG
        
        # 1. Chuyển Logits thành Allocation Map
        # allocation_map[rbg_idx] = ue_index (0..64), trong đó 0..63 là UE, 64 là None
        selected_indices = np.argmax(action, axis=1)
        
        # 2. Tính PRB Counts để gửi về MATLAB 
        # (Vì MATLAB hiện tại dùng performRBGMapping dựa trên số lượng PRB)
        # Chúng ta đếm số lần mỗi UE được chọn
        ue_counts = np.zeros(self.max_ues, dtype=int)
        for idx in selected_indices:
            if idx < self.max_ues: # Nếu không phải là "No Allocation"
                # Giả sử 1 RBG có kích thước chuẩn (ví dụ 16 PRB)
                # Ta cần nhân với RBG Size để ra số PRB, hoặc gửi số RBG.
                # Ở đây ta gửi số RBG counts, MATLAB cần nhân với rbgSize.
                ue_counts[idx] += 1
        
        # Gửi về MATLAB (chuyển đổi thành PRB counts ước lượng)
        # Lưu ý: MATLAB code của bạn đang mong đợi 'prbs' là số PRB, không phải số RBG.
        # Bạn nên cập nhật MATLAB để nhận 'rbg_counts' hoặc nhân ở đây.
        # Giả sử RBG Size = 16 (default 5G 100MHz)
        est_prb_counts = ue_counts * 16 
        
        if not self._send_to_matlab({"prbs": est_prb_counts.tolist()}):
            return self._process_obs(self.last_raw_state), 0, True, False, {}

        # 3. Nhận State mới
        next_raw_state = self._recv_from_matlab()
        if next_raw_state is None:
            return self._process_obs(self.last_raw_state), 0, True, False, {}
            
        self.last_raw_state = next_raw_state
        obs = self._process_obs(next_raw_state)
        
        # 4. Tính Reward (Giản lược)
        served = np.array(next_raw_state.get('last_served', []))
        reward = np.sum(np.log(served + 1e-6)) # Proportional Fair Proxy
        
        return obs, float(reward), False, False, {}

    def _process_obs(self, raw):
        # Raw features từ MATLAB: (MaxUEs, 5 + 2*Subbands)
        # Đảm bảo khớp shape (MaxUEs, feature_dim)
        feats = np.array(raw['features'], dtype=np.float32)
        
        # Padding hoặc cắt nếu số subbands không khớp config
        current_dim = feats.shape[1]
        target_dim = self.feature_dim
        
        if current_dim < target_dim:
            # Pad với 0
            padding = np.zeros((self.max_ues, target_dim - current_dim), dtype=np.float32)
            feats = np.hstack([feats, padding])
        elif current_dim > target_dim:
            # Cắt bớt
            feats = feats[:, :target_dim]
            
        return feats

    def _recv_from_matlab(self):
        # (Giữ nguyên logic nhận JSON từ code cũ)
        buffer = ""
        while "\n" not in buffer:
            try:
                data = self.conn.recv(4096)
                if not data: return None
                buffer += data.decode('utf-8')
            except: return None
        return json.loads(buffer.split("\n")[0])

    def _send_to_matlab(self, data):
        try:
            self.conn.sendall((json.dumps(data) + "\n").encode('utf-8'))
            return True
        except: return False