import os
import socket
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from matlab_training_env import MatlabBridgeEnv

SAVE_DIR = "runs/matlab_ppo_logging"
os.makedirs(SAVE_DIR, exist_ok=True)

class TensorboardCallback(BaseCallback):
    """Callback to log KPI metrics during training."""
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        if "cell_tput_Mb" in infos:
            self.logger.record("custom/cell_tput_Mb", infos["cell_tput_Mb"])
        if "jain" in infos:
            self.logger.record("custom/jain_index", infos["jain"])
        return True

class SaveOnEpisodeEndCallback(BaseCallback):
    """Callback to automatically save model at end of each episode."""
    def __init__(self, save_dir, verbose=0):
        super(SaveOnEpisodeEndCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            self.episode_count += 1
            path = os.path.join(self.save_dir, f"model_episode_{self.episode_count}")
            self.model.save(path)
            print(f"[SAVE] Episode {self.episode_count} completed. Model saved to {path}")
        return True

class StopOnMatlabFinishedCallback(BaseCallback):
    """Callback to stop training when MATLAB simulation finishes."""
    def _on_step(self) -> bool:
        try:
            env = self.model.get_env()
            # Access the underlying MatlabBridgeEnv through Monitor wrapper
            matlab_env = env.envs[0].env
            if matlab_env.matlab_finished:
                print("[STOP] MATLAB simulation finished. Stopping training immediately...")
                return False  # Stop training
        except Exception as e:
            print(f"[WARNING] Error checking MATLAB status: {e}")
        return True

def make_env():
    env = MatlabBridgeEnv()
    # Note: Using Monitor here, will be wrapped with VecMonitor at higher level
    monitor_path = os.path.join(SAVE_DIR, "monitor.csv")
    env = Monitor(env, filename=monitor_path, info_keywords=('jain','cell_tput_Mb','wasted_prbs'))
    return env

def main():
    print("[INFO] Starting PPO Training (CPU Mode with Auto-Save)...")
    
    logger = configure(SAVE_DIR, ["stdout", "csv", "tensorboard"])
    
    env = DummyVecEnv([make_env])
    env = VecMonitor(env) 
    
    # PPO configuration optimized for CPU execution with small MLP
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=3e-4, 
                batch_size=64, 
                n_steps=512,  
                ent_coef=0.01,
                tensorboard_log=SAVE_DIR,
                device='cpu') 
    
    model.set_logger(logger)
    
    # Callback configuration
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=SAVE_DIR, name_prefix="ppo_backup")
    episode_callback = SaveOnEpisodeEndCallback(save_dir=SAVE_DIR)
    log_callback = TensorboardCallback()
    stop_callback = StopOnMatlabFinishedCallback()
    
    my_callbacks = CallbackList([checkpoint_callback, episode_callback, log_callback, stop_callback])
    
    print("[WAIT] Waiting for MATLAB connection...")
    
    try:
        model.learn(total_timesteps=100_000, callback=my_callbacks)
        model.save(f"{SAVE_DIR}/final_model")
        print("[SUCCESS] Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Keyboard interrupt detected!")
        try:
            model.save(f"{SAVE_DIR}/interrupted_model")
            print("[SAVE] Model saved: interrupted_model.zip")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
    
    except socket.timeout:
        print("\n[TIMEOUT] Socket timeout - MATLAB CSI RS processing is taking longer than expected!")
        print("[INFO] Consider further increasing timeout if CSI RS processing requires more time.")
        try:
            model.save(f"{SAVE_DIR}/timeout_model")
            print("[SAVE] Model saved: timeout_model.zip")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            model.save(f"{SAVE_DIR}/error_model")
            print("[SAVE] Model saved: error_model.zip")
        except Exception as save_err:
            print(f"[ERROR] Failed to save model: {save_err}")
    
    finally:
        try:
            # Access the underlying MatlabBridgeEnv through Monitor wrapper
            matlab_env = env.envs[0].env
            if matlab_env.conn:
                matlab_env.conn.close()
            if matlab_env.sock:
                matlab_env.sock.close()
            print("[CLEANUP] Socket connections closed.")
        except Exception as e:
            print(f"[WARNING] Error during cleanup: {e}")

if __name__ == "__main__":
    main()