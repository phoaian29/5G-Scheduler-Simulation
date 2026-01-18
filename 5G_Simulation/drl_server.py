import socket
import json
import random
import time
import numpy as np
import sys

HOST = '127.0.0.1'
PORT = 5555

def get_test_action(state_dict):
    """
    Xử lý RAW DATA từ MATLAB
    State dict contains: 'WidebandFeatures' [NumUE x 5]
    Format: [Buffer(Bytes), Tput(bps), CQI, Rank, AllocRatio]
    """
    ue_list = state_dict.get('UE_RNTIs', [])
    features = np.array(state_dict.get('WidebandFeatures', [])) 
    
    num_rbg = 17 
    try:
        if 'CQI_Detail' in state_dict:
            cqis = state_dict['CQI_Detail']
            if isinstance(cqis, list) and len(cqis) > 0:
                num_rbg = len(cqis[0])
    except:
        pass

    if len(ue_list) == 0:
        return [0] * num_rbg

    print("\n" + "="*80, flush=True)
    print(f"[PYTHON RAW] Layer: {state_dict.get('Layer')} | PrevReward: {state_dict.get('PreviousReward')}", flush=True)
    print(f"{'UE_ID':<6} | {'Buffer(Bytes)':<14} | {'Tput(bps)':<14} | {'CQI':<6} | {'Rank':<6} | {'Alloc'}", flush=True)
    print("-" * 80, flush=True)
    
    for i, ue_id in enumerate(ue_list):
        f = features[i]
        print(f"{ue_id:<6} | {f[0]:<14.0f} | {f[1]:<14.2e} | {f[2]:<6.2f} | {f[3]:<6.0f} | {f[4]:.2f}", flush=True)
    print("=" * 80, flush=True)
    

    has_data_indices = np.where(features[:, 0] > 0)[0]
    
    if len(has_data_indices) > 0:
        cqi_of_active_ues = features[has_data_indices, 2]
        best_local_idx = np.argmax(cqi_of_active_ues)
        best_ue_idx = has_data_indices[best_local_idx]
        best_ue = ue_list[best_ue_idx]
        print(f"[DECISION] Priority UE: {best_ue} (Has Data & Max CQI)", flush=True)
    else:
        best_ue_idx = np.argmax(features[:, 2])
        best_ue = ue_list[best_ue_idx]
        print(f"[DECISION] Probing UE: {best_ue} (Best Channel)", flush=True)
    
    if random.random() > 0.3: 
        action = [best_ue] * num_rbg
    else:
        action = [random.choice(ue_list) for _ in range(num_rbg)]
        
    print(f"[SENDING] Action: {action[:5]}...", flush=True)
    return action

def start_server():
    print(f"[PYTHON] DRL Server (RAW MODE) starting on {HOST}:{PORT}...", flush=True)
    
    while True: 
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((HOST, PORT))
                except OSError:
                    print("Port busy, waiting 2s...", flush=True)
                    time.sleep(2)
                    continue
                    
                s.listen()
                print("[PYTHON] Waiting for MATLAB...", flush=True)
                
                conn, addr = s.accept()
                with conn:
                    print(f"[PYTHON] Connected: {addr}", flush=True)
                    buffer = ""
                    
                    while True:
                        data = conn.recv(16384) 
                        if not data: 
                            print("Connection closed.", flush=True)
                            break
                        
                        buffer += data.decode('utf-8')
                        
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if not line.strip(): continue
                            
                            try:
                                state = json.loads(line)
                                action = get_test_action(state)
                                
                                resp = json.dumps({"action": action})
                                conn.sendall(resp.encode('utf-8'))
                                conn.sendall(b'\n')
                                
                            except json.JSONDecodeError:
                                print("JSON Error", flush=True)
                                
        except Exception as e:
            print(f"Server Crash: {e}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    start_server()