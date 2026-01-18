import numpy as np

def jain_fairness(throughputs):
    """Compute Jain's Fairness Index."""
    n = len(throughputs)
    sum_t = np.sum(throughputs)
    sum_sq = np.sum(np.square(throughputs))
    if sum_sq == 0: return 0.0
    return (sum_t ** 2) / (n * sum_sq)

def bytes_per_prb(mcs_index):
    """Compute number of Bytes per PRB based on MCS (approximate 5G)."""
    # Approximate spectral efficiency table for MCS 0-28
    effs = [0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.40, 2.73, 
            3.32, 3.90, 4.52, 5.12, 5.55, 6.07, 6.23, 6.50, 6.70, 6.90, 
            7.00, 7.10, 7.20, 7.30, 7.35, 7.40, 7.45, 7.48, 7.50]
    
    if mcs_index < 0: mcs_index = 0
    if mcs_index > 28: mcs_index = 28
    
    efficiency = effs[mcs_index]
    # 1 PRB = 12 subcarriers * 14 symbols. Approx. 10% overhead
    resource_elements = 12 * 14 * 0.9 
    bits = efficiency * resource_elements
    return bits / 8.0  # Convert to Bytes

def project_scores_to_prbs(scores, budget, backlog, mcs, active_mask, training_mode=True):
    """Allocate the PRB budget according to scores produced by the agent.
    - scores: Output from the actor network (0.0 - 1.0)
    - budget: Total available PRBs (e.g., 51)
    - backlog: Pending data per UE (Bytes)
    - mcs: Channel quality (MCS indices)
    - active_mask: Indicator for UEs with data (1) or not (0)
    """
    num_ue = len(scores)
    
    # 1. Filter inactive UEs (Backlog = 0)
    # If backlog == 0, zero out the score to avoid wasteful allocation
    valid_scores = scores * active_mask
    
    # 2. Compute maximum demand of each UE (in PRBs)
    demands = np.zeros(num_ue)
    for i in range(num_ue):
        if active_mask[i] > 0:
            bpp = bytes_per_prb(mcs[i])
            if bpp > 0:
                demands[i] = np.ceil(backlog[i] / bpp)
            else:
                demands[i] = 1.0 # Avoid division by zero
    
    # 3. Allocate proportionally to scores (Weighted Proportional Fair)
    sum_scores = np.sum(valid_scores)
    allocation = np.zeros(num_ue, dtype=int)
    
    if sum_scores > 1e-6:
        # Initial proportional allocation
        raw_alloc = (valid_scores / sum_scores) * budget
        allocation = np.floor(raw_alloc).astype(int)
        
        # 4. Adjust: Do not allocate beyond demand (simple water-filling)
        # If allocation exceeds demand -> reduce it and return to the remaining pool
        for i in range(num_ue):
            if allocation[i] > demands[i]:
                allocation[i] = demands[i]
        
        # 5. Allocate remaining PRBs
        current_total = np.sum(allocation)
        remainder = budget - current_total
        
        # Prioritize distributing remainder to highest-score UEs that are still under demand
        if remainder > 0:
            # Sort indices by descending score
            sorted_idx = np.argsort(valid_scores)[::-1]
            for idx in sorted_idx:
                if remainder <= 0: break
                if active_mask[idx] and allocation[idx] < demands[idx]:
                    needed = demands[idx] - allocation[idx]
                    give = min(remainder, needed)
                    allocation[idx] += give
                    remainder -= give
                    
            # If there is still remainder (all demands satisfied), give to highest-score UE (to boost throughput)
            if remainder > 0:
                best_ue = np.argmax(valid_scores)
                allocation[best_ue] += remainder

    wasted_prbs = budget - np.sum(allocation)
    return allocation, demands, wasted_prbs, sum_scores