from dataclasses import dataclass

@dataclass
class DRLConfig:
    host: str = "0.0.0.0"
    port: int = 5555

    max_ues: int = 16          # MUST match SchedulerDRL.MaxUEs
    subband_size_prb: int = 16 # MUST match SchedulerDRL.SubbandSize

    # DSACD hyperparams
    n_quantiles: int = 16
    gamma: float = 0.0          # in paper they often use 0 for per-decision PF reward
    tau: float = 0.005
    beta_entropy: float = 0.98

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4

    replay_size: int = 200_000
    batch_size: int = 256
    warmup_steps: int = 5
    update_every: int = 1
    updates_per_step: int = 1

    # PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 200_000
