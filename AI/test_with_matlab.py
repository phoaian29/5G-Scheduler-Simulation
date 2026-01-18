import os
import time
import numpy as np
from stable_baselines3 import PPO
from matlab_training_env import MatlabBridgeEnv

# MODEL FILE PATH (Adjust filename if needed)
# Example: runs/matlab_ppo_logging/final_model.zip
# Or: runs/matlab_ppo_logging/interrupted_model.zip
MODEL_PATH = "runs/matlab_ppo_logging/interrupted_model.zip" 

def main():
    print("🚀 Starting Inference Mode (No Training)...")
    
    env = MatlabBridgeEnv()
    
    if os.path.exists(MODEL_PATH):
        print(f"📂 Loading model from: {MODEL_PATH}")
        # device='cpu' để đồng bộ với lúc train
        # device='cpu' to match training
        model = PPO.load(MODEL_PATH, env=env, device='cpu')
    else:
        print(f"❌ Model NOT found at {MODEL_PATH}")
        print("Please train the model first!")
        return
        
    print("⏳ Waiting for MATLAB connection...")
    
    obs, _ = env.reset()
    step_count = 0
    total_tput = 0
    
    try:
        while True:
            # deterministic=True: select the most deterministic (best) action
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            tput = info.get('cell_tput_Mb', 0)
            total_tput += tput
            
            # Print each step
            print(f"Step {step_count:04d} | Action: {action.round(2)} | Tput: {tput:.3f} Mb")

            if terminated or truncated:
                print("🏁 Episode finished. Resetting...")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped. Avg Tput: {total_tput/max(1, step_count):.3f} Mb/TTI")

if __name__ == "__main__":
    main()