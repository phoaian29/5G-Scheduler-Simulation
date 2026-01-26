import socket
import json
import numpy as np

class PPOTrainer:
    def __init__(self, port=5555):
        self.port = port
        self.server_socket = None
        self.client_socket = None
        
    def start_server(self):
        """Start TCP server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('127.0.0.1', self.port))
        self.server_socket.listen(1)
        
        print(f"🚀 [PYTHON] PPO Trainer listening on port {self.port}...")
        self.client_socket, addr = self.server_socket.accept()
        print(f"✅ [PYTHON] Connected from {addr}")
        
    def receive_and_process(self):
        """Receive CSI data from MATLAB"""
        buffer = b''
        
        while True:
            data = self.client_socket.recv(4096)
            if not data:
                break
                
            buffer += data
            
            if b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                
                try:
                    payload = json.loads(line.decode('utf-8'))
                    
                    # Parse CSI data
                    csi_data = self.parse_csi_data(payload)
                    
                    # Compute features (including cross-correlation)
                    state = self.compute_state(csi_data, payload)
                    
                    # DRL decision (dummy for now)
                    prbs = self.drl_decision(state, payload)
                    
                    # Send response
                    response = {'prbs': prbs.tolist()}
                    self.send_response(response)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    response = {'prbs': [0] * 64}
                    self.send_response(response)
    
    def parse_csi_data(self, payload):
        """Parse raw CSI data from MATLAB"""
        raw_csi = payload['raw_csi']
        
        csi_data = {
            'rank': np.array(raw_csi['rank']),
            'wideband_cqi': np.array(raw_csi['wideband_cqi']),
            'subband_cqi': np.array(raw_csi['subband_cqi']),
            'sinr_eff': np.array(raw_csi['sinr_eff']),
            'buffer': np.array(raw_csi['buffer']),
            'precoding': {},
            'pmi': {}
        }
        
        # Parse precoding matrices
        for p in payload['precoding']:
            ue_id = p['ue_id']
            real_part = np.array(p['real'])
            imag_part = np.array(p['imag'])
            
            # *** DEBUG: In ra giá trị thực tế ***
            if ue_id <= 3:
                print(f"\n[PARSE] UE {ue_id}:")
                print(f"  Shape: {real_part.shape}")
                print(f"  Real[0, 0:5, 0]: {real_part[0, 0:5, 0]}")
                print(f"  Imag[0, 0:5, 0]: {imag_part[0, 0:5, 0]}")
                print(f"  Norm: {np.linalg.norm(real_part[0, :, 0] + 1j * imag_part[0, :, 0]):.4f}")
            
            # Reconstruct complex matrix
            W_3D = real_part + 1j * imag_part
            csi_data['precoding'][ue_id] = W_3D
        
        # Parse PMI
        for pmi in payload['pmi']:
            ue_id = pmi['ue_id']
            csi_data['pmi'][ue_id] = {
                'i1': pmi.get('i1'),
                'i2': pmi.get('i2')
            }
        
        return csi_data
    
    def compute_state(self, csi_data, payload):
        """Compute full state including cross-correlation"""
        num_ues = 64
        num_subbands = payload['num_subbands']
        
        # Basic features from MATLAB
        features = np.array(payload['features'])  # [num_ues x 5]
        
        # Subband CQI features
        cqi_features = csi_data['subband_cqi']  # [num_ues x num_subbands]
        
        # *** COMPUTE CROSS-CORRELATION IN PYTHON ***
        rho_features = self.compute_cross_correlation(
            csi_data['precoding'], 
            payload['scheduled_per_rbg'], 
            num_subbands,
            num_ues
        )
        
        # Concatenate: [num_ues x (5 + 2*num_subbands)]
        state = np.concatenate([
            features,           # [num_ues x 5]
            cqi_features,       # [num_ues x num_subbands]
            rho_features        # [num_ues x num_subbands]
        ], axis=1)
        
        print(f"\n[ITER {payload['iteration']}] State shape: {state.shape}")
        print(f"  Eligible UEs: {payload['eligible_ues']}")
        print(f"  Scheduled UEs: {payload['scheduled_ues']}")
        print(f"  Rho stats: min={rho_features.min():.4f}, max={rho_features.max():.4f}, mean={rho_features.mean():.4f}")
        
        return state
    
    def compute_cross_correlation(self, precoding_dict, scheduled_per_rbg, num_subbands, num_ues):
        """Tính cross-correlation từ precoding matrices"""
        
        # *** DEBUG ***
        print(f"\n[DEBUG CORR]")
        print(f"  Precoding dict contains UEs: {list(precoding_dict.keys())}")
        print(f"  Scheduled per RBG: {[(item['rbg_idx'], item['ue_list']) for item in scheduled_per_rbg[:3]]}")
        
        rho_matrix = np.zeros((num_ues, num_subbands))
        
        # Convert to dict
        scheduled_dict = {}
        for item in scheduled_per_rbg:
            rbg_idx = item['rbg_idx']
            ue_list = item['ue_list']
            
            if ue_list is None or ue_list == []:
                scheduled_dict[rbg_idx] = []
            elif isinstance(ue_list, list):
                scheduled_dict[rbg_idx] = [int(u) for u in ue_list]
            elif isinstance(ue_list, (int, float)):
                scheduled_dict[rbg_idx] = [int(ue_list)]
            else:
                scheduled_dict[rbg_idx] = []
        
        # *** DEBUG ***
        print(f"  Converted scheduled_dict (first 3): {dict(list(scheduled_dict.items())[:3])}")
        
        # Compute correlation
        computed_count = 0
        for ue_id, W_3D in precoding_dict.items():
            for sb_idx in range(num_subbands):
                rbg_idx = sb_idx + 1
                
                P_mu = W_3D[:, :, sb_idx]
                
                scheduled_ues = scheduled_dict.get(rbg_idx, [])
                if not scheduled_ues:
                    rho_matrix[ue_id-1, sb_idx] = 0.0
                    continue
                
                max_corr = 0.0
                for scheduled_ue in scheduled_ues:
                    if scheduled_ue not in precoding_dict:
                        # *** DEBUG ***
                        if computed_count < 5:
                            print(f"  [SKIP] UE {scheduled_ue} not in precoding_dict")
                        continue
                    
                    W_sc = precoding_dict[scheduled_ue]
                    P_mc = W_sc[:, :, sb_idx]
                    
                    cross_prod = P_mu @ P_mc.conj().T
                    corr = np.max(np.abs(cross_prod))
                    max_corr = max(max_corr, corr)
                    
                    # *** DEBUG ***
                    if computed_count < 5:
                        print(f"  [COMP] UE {ue_id} vs UE {scheduled_ue} @ RBG {rbg_idx}: corr={corr:.4f}")
                        computed_count += 1
                
                rho_matrix[ue_id-1, sb_idx] = max_corr
        
        return rho_matrix
    
    def drl_decision(self, state, payload):
        """DRL decision (dummy - uniform allocation for now)"""
        num_ues = 64
        prb_budget = payload['prb_budget']
        eligible_ues = payload['eligible_ues']
        
        # Dummy: allocate equally among eligible UEs with buffer
        prbs = np.zeros(num_ues)
        
        buffer = state[:, 3]  # buffer status
        active_ues = [ue for ue in eligible_ues if buffer[ue-1] > 0]
        
        if active_ues:
            prb_per_ue = prb_budget // len(active_ues)
            for ue in active_ues:
                prbs[ue-1] = prb_per_ue
        
        return prbs
    
    def send_response(self, response):
        """Send response back to MATLAB"""
        response_str = json.dumps(response) + '\n'
        self.client_socket.sendall(response_str.encode('utf-8'))
    
    def close(self):
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()


if __name__ == '__main__':
    trainer = PPOTrainer(port=5555)
    try:
        trainer.start_server()
        trainer.receive_and_process()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        trainer.close()