# Deep Reinforcement Learning for 5G NR MAC Scheduling using MATLAB and Python Integration

## 1. Project Overview
This project implements a Deep Reinforcement Learning (DRL) based scheduler for the Medium Access Control (MAC) layer of a 5G New Radio (NR) downlink system. The architecture utilizes a hybrid simulation approach where MATLAB serves as the precise environment simulator—handling PHY/MAC layer physics, channel modeling, and numerology—and Python acts as the intelligent agent host implementing the Proximal Policy Optimization (PPO) algorithm.

The primary objective is to solve the resource allocation problem in 5G Downlink, specifically determining the optimal distribution of Physical Resource Blocks (PRBs) among multiple User Equipments (UEs) to maximize cell throughput while maintaining fairness.

## 2. System Architecture
The simulation environment is decoupled into two distinct subsystems communicating via a synchronous TCP/IP interface.

### 2.1. MATLAB Subsystem (Environment)
Located in the `5G_Simulation` directory, the MATLAB component utilizes the 5G Toolbox to simulate the physical and link layers. Its key responsibilities include:
* **Channel Modeling:** Generating realistic propagation conditions using CDL/TDL models.
* **CSI Feedback:** Calculating Channel Quality Indicator (CQI) and Rank Indicator (RI) based on signal measurements.
* **Buffer Management:** Tracking RLC buffer status for each UE.
* **Physical Layer Abstraction:** Mapping allocated Resource Block Groups (RBGs) to transport blocks and calculating the resulting throughput based on Block Error Rate (BLER).
* **Client Interface:** Acts as a TCP Client (via `SchedulerDRL.m`), transmitting state observations to Python and receiving scheduling actions.

### 2.2. Python Subsystem (Agent & Training)
Located in the `AI` directory, the Python component hosts the DRL agent and the resource distribution logic. Its key responsibilities include:
* **RL Algorithm:** Implements Proximal Policy Optimization (PPO) using the `stable-baselines3` library.
* **Action Processing:** Converts continuous probability scores from the Neural Network into discrete PRB allocations using a water-filling algorithm.
* **Server Interface:** Acts as a TCP Server, listening for connections from MATLAB.

## 3. Reinforcement Learning Formulation

### 3.1. Observation Space (State)
The state space represents the instantaneous condition of the network. The input to the Neural Network is a flattened vector of size `[N_UEs x 5]`. For a standard 4-UE scenario, the input dimension is 20.

For each User Equipment (UE) *i*, the features are defined as follows:
1.  **Buffer Status (Normalized):** The amount of data pending in the RLC buffer, normalized by the maximum estimated capacity.
2.  **Average Throughput (Normalized):** Exponential Moving Average (EMA) of the UE's throughput, normalized by the maximum possible link rate.
3.  **Wideband CQI (Normalized):** The average Channel Quality Indicator over the entire bandwidth (range 0-15), normalized to [0, 1].
4.  **Rank Indicator (Normalized):** The MIMO rank (range 1-4), normalized to [0, 1].
5.  **Allocation Ratio (Previous TTI):** The ratio of resources allocated to this UE in the previous transmission time interval.

### 3.2. Action Space
The action space is continuous, designed to output priority scores rather than direct resource blocks to reduce dimensionality.
* **Type:** Continuous (Box)
* **Range:** [0.0, 1.0]
* **Dimension:** Vector of size *N* (where *N* is the number of UEs).
* **Interpretation:** Each value represents the scheduling priority weight for the corresponding UE.

### 3.3. Resource Allocation Logic (Translation Layer)
Since the PPO agent outputs continuous weights, a translation layer (implemented in `AI/rl_mac_env.py`) converts these weights into integer PRB counts. The logic follows a **Weighted Proportional Fair** mechanism with **Water-filling**:

1.  **Raw Allocation:** Resources are divided proportionally based on the action weights relative to the sum of all weights.
2.  **Demand Constraint:** The allocation is capped by the UE's actual buffer demand to prevent resource wastage.
3.  **Redistribution:** Any PRBs unused due to low demand from high-priority UEs are redistributed to other UEs with remaining backlog.

This ensures the system adheres to the physical bandwidth constraints (e.g., 51 PRBs) and optimizes spectral efficiency by preventing allocation to empty buffers.

### 3.4. Reward Function
The objective function balances throughput maximization and user fairness.

Reward = alpha * (Total Cell Throughput / Max Cell Capacity) + beta * Jain's Fairness Index

Where:
* **Jain's Fairness Index** is calculated based on the averaged throughput of all users.
* **alpha** and **beta** are weighting coefficients (default: alpha=1.0, beta=0.2).

## 4. Communication Protocol
The data exchange follows a synchronous request-response model over TCP sockets using JSON serialization.

### 4.1. Handshake
1.  Python opens a socket server on `127.0.0.1:5555`.
2.  MATLAB connects to this address as a client.

### 4.2. Runtime Loop (Per TTI)
1.  **MATLAB (Send):** Sends a JSON packet containing the Observation Matrix (`features`), `last_served` bytes, and the current `prb_budget`.
2.  **Python (Receive):** Parses JSON, normalizes data, and feeds it to the PPO Policy Network.
3.  **Python (Process):** The network outputs Action Scores. The allocation logic computes the integer number of PRBs for each UE.
4.  **Python (Send):** Sends a JSON packet containing the PRB counts (e.g., `{"prbs": [10, 15, 20, 6]}`).
5.  **MATLAB (Receive):** Decodes the PRB counts.
6.  **MATLAB (Execute):** The `SchedulerDRL` class maps these counts to specific Resource Block Groups (RBGs) in the frequency domain and schedules the transmission.

## 5. Directory Structure and File Descriptions

### Python Components (Directory: `/AI`)
* **`matlab_training_env.py`**: Defines the custom Gymnasium environment. Handles socket communication, state normalization, and reward calculation. Includes graceful shutdown and auto-save logic.
* **`rl_mac_env.py`**: Contains the mathematical logic for converting Action Scores into PRB integers (Water-filling algorithm) and calculating fairness metrics.
* **`train_with_matlab.py`**: The main entry point for training. Configures the PPO agent, sets up callbacks for logging and checkpointing, and initiates the training loop.
* **`test_with_matlab.py`**: Script for inference/testing. Loads a trained model and runs it deterministically without updating weights.
* **`visualize_results.py`**: Utility to parse training logs (`progress.csv`) and generate convergence plots.

### MATLAB Components (Directory: `/5G_Simulation`)
* **`SchedulerDRL.m`**: Custom scheduler class inheriting from `nrScheduler`. It overrides the scheduling strategy to fetch decisions from the Python server instead of using internal heuristics.
* **`NRCellPerformanceWithDownlinkMUMIMOExample.mlx`**: The main simulation script that configures the gNB, UEs, traffic models, and initiates the simulation loop.
* **`nrScheduler.m`, `nrMAC.m`, etc.**: Supporting classes for the 5G NR protocol stack.

## 6. Installation and Prerequisites

### 6.1. MATLAB Requirements
* MATLAB R2021a or later.
* 5G Toolbox.
* Instrument Control Toolbox (required for `tcpclient`).

### 6.2. Python Requirements
* Python 3.8 or later.
* Required packages:
  * `numpy`
  * `gymnasium`
  * `stable-baselines3`
  * `pandas`
  * `matplotlib`
  * `shimmy`

## 7. Execution Instructions

The system must be started in a specific order to establish the TCP connection correctly.

### Step 1: Start the Python Server
Open a terminal in the `/AI` directory and execute the training script. This will initialize the model and wait for a client connection.

```bash
cd AI
pythonExpected Output: [TRAINER] Waiting for MATLAB at 127.0.0.1:5555...

### Step 2: Start the MATLAB Simulation
Open MATLAB, navigate to the /5G_Simulation directory, and run the main simulation script (e.g., NRCellPerformanceWithDownlinkMUMIMOExample.mlx). Ensure the scheduler configuration is set to use SchedulerDRL.

Expected Output: [MATLAB] Connected!

### Step 3: Monitoring
Real-time: The Python console will log step-by-step rewards, PRB allocations, and buffer status every 10 steps.

Logging: Training metrics are saved to AI/runs/matlab_ppo_logging/.

Auto-Save: The model is automatically saved upon completion of a simulation episode or if interrupted via Keyboard Interrupt.

## 8. Technical Notes
Synchronization: The simulation runs in lock-step. MATLAB pauses execution while waiting for the Python response. This ensures data consistency but limits simulation speed to the sum of TCP overhead and Python inference time.

Dynamic Configuration: The prb_budget is dynamically synchronized. If the MATLAB bandwidth configuration changes, the Python environment automatically adjusts.

Device: CPU execution is enforced for the PPO agent to minimize latency associated with GPU data transfer for small-scale MLP networks. train_with_matlab.py
