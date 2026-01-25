import socket
import json
import numpy as np

HOST = '127.0.0.1'
PORT = 5555
MAX_UES = 64
DEFAULT_RBG_SIZE = 16 

def run_debug_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    print(f"🚀 [PYTHON] Đang chờ MATLAB tại {HOST}:{PORT}...")
    conn, addr = server_socket.accept()
    print(f"✅ MATLAB đã kết nối: {addr}\n")
    
    conn.settimeout(1.0)
    buffer = ""
    
    try:
        while True:
            try:
                data = conn.recv(32768)
                if not data: break
                buffer += data.decode('utf-8')
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip(): continue
                    
                    state = json.loads(line)
                    features = np.array(state.get('features', []))
                    num_sb = int(state.get('num_subbands', 18))
                    
                    # Tìm UE đầu tiên có Buffer > 0 để so sánh
                    active_ues = np.where(features[:, 3] > 0)[0]
                    
                    if len(active_ues) > 0:
                        u = active_ues[0]
                        vec = features[u]
                        
                        # --- BÓC TÁCH 7 FEATURES (CORRECT LABELS) ---
                        f_R = vec[0]
                        f_h = vec[1]
                        f_d = vec[2] # <--- ĐÂY LÀ ALLOCATED RBs (Normalized)
                        f_b = vec[3]
                        f_o = vec[4]
                        
                        start_sb = 5
                        end_sb = 5 + num_sb
                        f_g_vec = vec[start_sb : end_sb]
                        f_rho_vec = vec[end_sb :]
                        
                        print(f"--- [PYTHON RECEIVED UE {u+1}] ---")
                        print(f"1. Tput (f_R):     {f_R:.4f}")
                        print(f"2. Rank (f_h):     {f_h:.4f}")
                        print(f"3. AllocRBs (f_d): {f_d:.4f} (Corrected Label)") 
                        print(f"4. Buffer (f_b):   {f_b:.0f}")
                        print(f"5. WB CQI (f_o):   {f_o:.4f}")
                        print(f"6. SB CQI (Vec):   [{f_g_vec[0]:.2f}, {f_g_vec[1]:.2f}.. size={len(f_g_vec)}]")
                        print(f"7. Corr (Vec):     [{f_rho_vec[0]:.2f}..]")
                        print(f"---------------------------------")
                    
                    # --- DUMMY RESPONSE ---
                    # Logic random nhưng ưu tiên UE có buffer
                    num_subbands = state.get('num_subbands', 18)
                    action_dim = MAX_UES + 1
                    
                    smart_logits = np.full((num_subbands, action_dim), -1e9)
                    smart_logits[:, MAX_UES] = 0.0 # Default No-Op
                    
                    if len(active_ues) > 0:
                        rand_vals = np.random.randn(num_subbands, len(active_ues))
                        smart_logits[:, active_ues] = rand_vals + 2.0
                        
                    allocation_map = np.argmax(smart_logits, axis=1)
                    
                    ue_counts = np.zeros(MAX_UES, dtype=int)
                    for x in allocation_map:
                        if x < MAX_UES: ue_counts[x] += 1
                    
                    response = {"prbs": (ue_counts * DEFAULT_RBG_SIZE).tolist()}
                    conn.sendall((json.dumps(response) + "\n").encode('utf-8'))
                    
            except socket.timeout:
                continue
    except KeyboardInterrupt:
        print("\n🛑 Dừng Server.")
    finally:
        if conn: conn.close()
        server_socket.close()

if __name__ == "__main__":
    run_debug_server()