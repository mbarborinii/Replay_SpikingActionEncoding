import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint



# ----------------------------
# 1. Dataset: Oscillatory signals
# ----------------------------
class MassOscillationDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=500, dt=0.1,
                 m1=1.0, m2=2.0, k1=2.0, k2=1.0, k3=0.5, z_normalization=False):
        self.data = []
        self.seq_len = seq_len
        self.dt = dt
        self.time = np.arange(0, seq_len * dt, dt)

        for _ in range(n_samples):
            # Randomize initial displacements (you can also randomize masses or stiffness if needed)
            x1i, x2i = np.random.uniform(-1, 1, size=2)
            v1i, v2i = np.zeros(2)
            
            init = [x1i, x2i, v1i, v2i]
            sol = odeint(self.equations, init, self.time, args=(m1, m2, k1, k2, k3))

            x = sol[:, :2]  # only positions of masses
            self.data.append(torch.tensor(x, dtype=torch.float32))  # shape: [seq_len, 2]
        
        self.data_original = self.data.copy() # keep original data for later use (e.g. for plotting)
            
        if z_normalization:
            all_data = torch.cat(self.data, dim=0)  # [n_samples*seq_len, 4]
            self.mean = all_data.mean(dim=0)
            self.std = all_data.std(dim=0)
            self.data = [(x - self.mean) / self.std for x in self.data]
          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # autoregression (input, target)

    @staticmethod
    def equations(X, t, m1, m2, k1, k2, k3):
        x1, x2, v1, v2= X
        dx1 = v1
        dx2 = v2
        dv1 = (- k1 * x1 - k2 * (x1 + x2  )) / m1  # - (( k1 + k2)/m1) * x1 - (k2/m1 )* x2
        dv2 = ( - k2 * (x1 + x2) - k3 * x2) / m2#- (k2/m2) * x1 - ((k2 + k3)/m2) * x2 
        return [dx1, dx2, dv1, dv2]



class MassOscillationDataset_with_velocities(Dataset):
    def __init__(self, n_samples=1000, seq_len=500, dt=0.1,
                 m1=1.0, m2=2.0, k1=2.0, k2=1.0, k3=0.5, margin = 5, x1_init = None, x2_init = None, z_normalization = False):
        self.data = []
        self.seq_len = seq_len
        self.dt = dt
        self.time = np.arange(0, (margin + seq_len) * dt, dt)

        for _ in range(n_samples):
            # Randomize initial displacements (you can also randomize masses or stiffness if needed)
            if x1_init is not None and x2_init is not None:
                x1i, x2i = x1_init, x2_init

            else:
                self.range = 0.25
                x1i, x2i = np.random.uniform(-self.range, self.range, size=2)
                self.RANDOM_CHECK = True
                
            v1i, v2i = np.zeros(2)
            init = [x1i, x2i, v1i, v2i]
            sol = odeint(self.equations, init, self.time, args=(m1, m2, k1, k2, k3))

            x = sol[:, :4]  # positions and velocities of masses
            
            self.data.append(torch.tensor(x[margin:, :], dtype=torch.float32))  # shape: [seq_len, 2]
        
        self.data_original = self.data.copy() # keep original data for later use (e.g. for plotting)
            
        if z_normalization:
            all_data = torch.cat(self.data, dim=0)  # [n_samples*seq_len, 4]
            self.mean = all_data.mean(dim=0)
            self.std = all_data.std(dim=0)
            self.data = [(x - self.mean) / self.std for x in self.data]
            
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # autoregression (input, target)
    
    def get_min_max(self, sample_idx=None, quantity_idx=None):

        # Case 1: specific sample
        if sample_idx is not None:
            x = self.data[sample_idx]  # [things, time]

            if quantity_idx is not None:
                x = x[quantity_idx]       # [time]

            return x.min(), x.max()

        # Case 2: all samples
        mins = []
        maxs = []

        for x in self.data:             # each x: [things, time]
            if quantity_idx is not None:
                x = x[quantity_idx]        # [time]

            mins.append(x.min())
            maxs.append(x.max())

        return torch.min(torch.stack(mins)), torch.max(torch.stack(maxs))


    @staticmethod
    def equations(X, t, m1, m2, k1, k2, k3):
        x1, x2, v1, v2= X
        dx1 = v1
        dx2 = v2
        dv1 = (- k1 * x1 + k2 * (x2 - x1 )) / m1  # - (( k1 + k2)/m1) * x1 - (k2/m1 )* x2
        dv2 = ( - k2 * (x2 - x1) - k3 * x2) / m2#- (k2/m2) * x1 - ((k2 + k3)/m2) * x2 
        return [dx1, dx2, dv1, dv2]




class MassOscillationDataset_with_forzante(Dataset):
    def __init__(self, n_samples=1000, seq_len=500, dt=0.1,
                 m1=1.0, m2=2.0, k1=2.0, k2=1.0, k3=0.5, lambd=1.0, mu = 1.0, x_ideal = 0.0, x1_init = None, x2_init = None, z_normalization = False, RANDOM_CHECK=False):
        self.data = []
        self.seq_len = seq_len
        self.dt = dt
        self.time = np.arange(0, seq_len * dt, dt)
        if RANDOM_CHECK:
            self.x_ideal = np.random.uniform(-x_ideal, x_ideal, size = n_samples)
        else: 
            self.x_ideal = x_ideal
        self.lambd = lambd  
        self.mu = 2.0 * np.sqrt((self.lambd + k2 + k3) * m2)  # critical damping for the second mass
        self.RANDOM_CHECK = RANDOM_CHECK

        for i in range(n_samples):
            # Randomize initial displacements (you can also randomize masses or stiffness if needed)
            if x1_init is not None and x2_init is not None:
                x1i, x2i = x1_init, x2_init

            else:
                self.range = 0.25
                x1i, x2i = np.random.uniform(-self.range, self.range, size=2)
                # there was a print message that required a check in the range of the initial displacement which required the user to press enter to continue
                
            v1i, v2i = np.zeros(2)

            init = [x1i, x2i, v1i, v2i]
            if self.RANDOM_CHECK:
                sol = odeint(self.equations, init, self.time, args=(m1, m2, k1, k2, k3, lambd, mu, self.x_ideal[i]))
            else:
                sol = odeint(self.equations, init, self.time, args=(m1, m2, k1, k2, k3, lambd, mu, self.x_ideal))

            x = sol[:, :2]  # positions and velocities of masses
            self.data.append(torch.tensor(x, dtype=torch.float32))  # shape: [seq_len, 2]

        self.data_original = self.data.copy() # keep original data for later use (e.g. for plotting)
            
        if z_normalization:
            all_data = torch.cat(self.data, dim=0)  # [n_samples*seq_len, 4]
            self.mean = all_data.mean(dim=0)
            self.std = all_data.std(dim=0)
            self.data = [(x - self.mean) / self.std for x in self.data]
          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # autoregression (input, target)

    @staticmethod
    def equations(X, t, m1, m2, k1, k2, k3, lambd, mu, x_ideal):
        x1, x2, v1, v2= X
        
        # Feedforward: spring forces that would act at (x1_eq, x_ideal)
        x1_eq = k2 / (k1 + k2) * x_ideal
        ff = k2 * (x_ideal - x1_eq) + k3 * x_ideal  # force needed to hold x2 at x_ideal
        
        dx1 = v1
        dx2 = v2
        dv1 = (- k1 * x1 + k2 * (x2 - x1 )) / m1  
        dv2 = ( - k2 * (x2 - x1) - k3 * x2) / m2 + lambd * (x_ideal - x2) + ff/m2 - mu*v2 
        # print(x_ideal, x2, (( - k2 * (x1 - x2) - k3 * x2) / m2 ) + lambd * (x_ideal - x2))
        return [dx1, dx2, dv1, dv2]



class MassOscillationDataset_with_vels_and_forzante(Dataset):
    def __init__(self, n_samples=1000, seq_len=500, dt=0.1,
                 m1=1.0, m2=2.0, k1=2.0, k2=1.0, k3=0.5, lambd=1.0, mu = 1.0, x_ideal = 0.0, x1_init = None, x2_init = None, z_normalization = False, RANDOM_CHECK=False):
        self.data = []
        self.seq_len = seq_len
        self.dt = dt
        self.time = np.arange(0, seq_len * dt, dt)
        if RANDOM_CHECK:
            self.x_ideal = np.random.uniform(-x_ideal, x_ideal, size = n_samples)
        else: 
            self.x_ideal = x_ideal
        self.lambd = lambd  
        self.mu = 2.0 * np.sqrt((self.lambd + k2 + k3) * m2)  # critical damping for the second mass
        self.RANDOM_CHECK = RANDOM_CHECK

        for i in range(n_samples):
            # Randomize initial displacements (you can also randomize masses or stiffness if needed)
            if x1_init is not None and x2_init is not None:
                x1i, x2i = x1_init, x2_init

            else:
                self.range = 0.25
                x1i, x2i = np.random.uniform(-self.range, self.range, size=2)
                # there was a print message that required a check in the range of the initial displacement which required the user to press enter to continue
                

            v1i, v2i = np.zeros(2)

            init = [x1i, x2i, v1i, v2i]
            if self.RANDOM_CHECK:
                sol = odeint(self.equations, init, self.time, args=(m1, m2, k1, k2, k3, lambd, mu, self.x_ideal[i]))
            else:
                sol = odeint(self.equations, init, self.time, args=(m1, m2, k1, k2, k3, lambd, mu, self.x_ideal))

            x = sol[:, :4]  # positions and velocities of masses
            self.data.append(torch.tensor(x, dtype=torch.float32))  # shape: [seq_len, 2]

        self.data_original = self.data.copy() # keep original data for later use (e.g. for plotting)
            
        if z_normalization:
            all_data = torch.cat(self.data, dim=0)  # [n_samples*seq_len, 4]
            self.mean = all_data.mean(dim=0)
            self.std = all_data.std(dim=0)
            self.data = [(x - self.mean) / self.std for x in self.data]
          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # autoregression (input, target)

    @staticmethod
    def equations(X, t, m1, m2, k1, k2, k3, lambd, mu, x_ideal):
        x1, x2, v1, v2= X
        
        # Feedforward: spring forces that would act at (x1_eq, x_ideal)
        x1_eq = k2 / (k1 + k2) * x_ideal
        ff = k2 * (x_ideal - x1_eq) + k3 * x_ideal  # force needed to hold x2 at x_ideal
        
        
        dx1 = v1
        dx2 = v2
        dv1 = (- k1 * x1 + k2 * (x2 - x1 )) / m1  
        dv2 = ( - k2 * (x2 - x1) - k3 * x2) / m2 + lambd * (x_ideal - x2) + ff/m2 - mu*v2 
        # print(x_ideal, x2, (( - k2 * (x1 - x2) - k3 * x2) / m2 ) + lambd * (x_ideal - x2))
        return [dx1, dx2, dv1, dv2]




class MassOscillationDataset_multipledesiredpos(Dataset):
    def __init__(self, n_samples=1000, tot_seq_len=500, single_seq_len=[250,250], dt=0.1,
                 m1=1.0, m2=2.0, k1=2.0, k2=1.0, k3=0.5, lambd=1.0, mu = 1.0, x_ideal = [0.0, 1.0], x1_init = None, x2_init = None, z_normalization = False):
        
        assert len(single_seq_len) == len(x_ideal), "Shape single sequence length must be the same as the number of final positions"
        assert np.sum(single_seq_len) == tot_seq_len, "The duration of the final positions summed must be equal to the total simulation time"
        
        self.data = []
        self.tot_seq_len = tot_seq_len
        self.single_seq_len = single_seq_len
        self.dt = dt
        self.time = np.arange(0, tot_seq_len * dt, dt)
        self.x_ideal = x_ideal
        self.lambd = lambd  
        self.mu = 2.0 * np.sqrt(self.lambd * m2)  # critical damping for the second mass

        for _ in range(n_samples):
            # Randomize initial displacements (you can also randomize masses or stiffness if needed)
            if x1_init is not None and x2_init is not None:
                x1i, x2i = x1_init, x2_init

            else:
                self.range = 0.25
                x1i, x2i = np.random.uniform(-self.range, self.range, size=2)
                self.RANDOM_CHECK = True
                # there was a print message that required a check in the range of the initial displacement which required the user to press enter to continue
                

            v1i, v2i = np.zeros(2) # I start static

            init = [x1i, x2i, v1i, v2i]
            sol = odeint(self.equations, init, self.time, args=(m1, m2, k1, k2, k3, lambd, mu, x_ideal, single_seq_len, dt))

            x = sol[:, :4]  # positions and velocities of masses
            self.data.append(torch.tensor(x, dtype=torch.float32))  # shape: [tot_seq_len, 2]

        self.data_original = self.data.copy() # keep original data for later use (e.g. for plotting)
            
        if z_normalization:
            all_data = torch.cat(self.data, dim=0)  # [n_samples*seq_len, 4]
            self.mean = all_data.mean(dim=0)
            self.std = all_data.std(dim=0)
            self.data = [(x - self.mean) / self.std for x in self.data]
          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]  # autoregression (input, target)

    @staticmethod
    def equations(X, t, m1, m2, k1, k2, k3, lambd, mu, x_ideal, single_seq_len, dt):
        
        x1, x2, v1, v2= X
        
        # Determine which segment we're in based on t
        cumulative_times = np.cumsum([s * dt for s in single_seq_len])
        x_desired = x_ideal[-1]  # default to last if t overshoots
        for i, cumul_t in enumerate(cumulative_times):
            if t < cumul_t:
                x_desired = x_ideal[i]
                break
        
        # Feedforward: spring forces that would act at (x1_eq, x_ideal)
        x1_eq = k2 / (k1 + k2) * x_desired
        ff = k2 * (x_desired - x1_eq) + k3 * x_desired  # force needed to hold x2 at x_ideal
        
        
        dx1 = v1
        dx2 = v2
        dv1 = (- k1 * x1 + k2 * (x2 - x1 )) / m1  
        dv2 = ( - k2 * (x2 - x1) - k3 * x2) / m2 + lambd * (x_desired - x2) + ff/m2 - mu*v2 
        # print(x_ideal, x2, (( - k2 * (x1 - x2) - k3 * x2) / m2 ) + lambd * (x_ideal - x2))
        return [dx1, dx2, dv1, dv2]




class TwoLinkArmDataset(Dataset):
    """
    Dataset simulating a 2-link planar robot arm under PD control,
    with encoder-like position readout (joint angles only).

    State vector: [q1, q2, dq1, dq2]
      - q1, q2   : joint angles (rad)
      - dq1, dq2 : joint velocities (rad/s)

    Dynamics (standard manipulator equation):
        M(q) * ddq + C(q, dq) * dq + g(q) = tau

    Control law (PD with optional gravity feedforward):
        tau = Kp * (q_des - q) - Kd * dq  [+ G(q_des)]

    The output stored is [q1(t), q2(t)] only, mimicking encoder readout.

    Motion structure
    ----------------
    Each sample is built from n_rep repetitions. One repetition = one
    forward leg (q_init → q_des) plus, if n_rep > 1, one return leg
    (q_des → q_init). Legs are simulated sequentially: the end state of
    each leg is the exact initial condition of the next, so velocity
    continuity is preserved.

    Leg boundaries are stored in self.leg_boundaries[i] (raw, pre-policy)
    and self.leg_boundaries_final[i] (after length policy). The checker
    always reads leg_boundaries_final so it stays in sync with the stored
    data regardless of same_length / sim_length settings.

    Convergence check
    -----------------
    A trajectory is "converged" if, at the end of every leg, all joints
    are within conv_delta (rad) of the leg's target. This is evaluated on
    the RAW simulated trajectory (before any length policy), so the check
    is always physically meaningful.

    Two parameters control what happens when convergence fails:

      require_convergence=False (default)
        All samples are kept regardless of convergence. Use this when
        you want the dataset to contain failures too (e.g. for training
        a network to generalise to real-world imperfection).

      require_convergence=True
        Non-converging samples are discarded and re-simulated with fresh
        random parameters (up to max_resample_attempts times). Use this
        for a "clean" dataset. If all attempts fail a warning is printed
        and the last attempt is kept.

    self.convergence_log[i] is a list of bools (one per leg) recording
    whether each leg converged, regardless of require_convergence.

    Stability guard
    ---------------
    Independently of convergence, if any joint angle exceeds angle_limit
    (rad) the trajectory is considered physically unstable and is always
    re-simulated (up to max_resample_attempts). This guard runs first and
    cannot be disabled.

    Speed control
    -------------
    Gesture speed is controlled via the desired settling time T_s (s):
        omega_n = 5.8 / T_s
        Kp      = m2 * omega_n^2
        Kd      = 2 * sqrt(m2 * Kp)   (critically damped)

    Fixed-length output (same_length)
    ----------------------------------
    When same_length=True every trajectory is resampled to sim_length
    steps via index interpolation after simulation.
    When same_length=False sim_length acts as a hard trim.
    Variable-length output (same_length=False, no sim_length) prints a
    warning because it is incompatible with DataLoader batching.

    Parameters
    ----------
    n_samples             : number of independent trajectories
    dt                    : timestep (s)
    n_rep                 : number of forward legs (repetitions).
                            1  → q_init → q_des
                            >1 → q_init → q_des → q_init → ... → q_des
    T_s                   : fixed settling time (s). If None, randomized
                            per sample from T_s_range.
    T_s_range             : (T_s_min, T_s_max) used when T_s is None.
    leg_time_factor       : each leg runs for leg_time_factor * T_s s.
    same_length           : if True resample to sim_length steps.
    sim_length            : output length in steps. Required when
                            same_length=True; acts as trim otherwise.
    angle_limit           : stability threshold (rad). Default pi.
    require_convergence   : if True, discard and re-simulate samples
                            where any leg fails to reach its target
                            within conv_delta. Default False.
    conv_delta            : convergence threshold (rad) per joint.
                            Used when require_convergence=True and
                            always logged in convergence_log.
                            Default 0.05 rad (~3 deg).
    max_resample_attempts : max re-simulation attempts per sample
                            (applies to both stability and convergence
                            guards). Default 10.
    l1, l2                : link lengths (m)
    m1, m2                : link masses (kg)
    g                     : gravitational acceleration (m/s^2)
    Kp                    : fixed PD proportional gain (overrides T_s).
    Kd                    : fixed PD derivative gain.
    gravity_ff            : if True add gravity feedforward to torque.
    q_des                 : fixed desired angles [q1_des, q2_des] (rad).
                            If None randomized from q_des_range.
    q_des_range           : (low, high) uniform range for q_des (rad).
    q_init                : fixed initial angles [q1_0, q2_0] (rad).
                            If None randomized from q_init_range.
    q_init_range          : (low, high) uniform range for q_init (rad).
    z_normalization       : if True standardize output trajectories.
    """

    def __init__(
        self,
        n_samples=1000,
        dt=0.01,
        n_rep=1,
        T_s=None,
        T_s_range=(0.5, 3.0),
        leg_time_factor=2.0,
        same_length=False,
        sim_length=None,
        angle_limit=np.pi,
        require_convergence=False,
        conv_delta=0.05,
        max_resample_attempts=10,
        l1=1.0,
        l2=0.8,
        m1=1.0,
        m2=0.5,
        g=9.81,
        Kp=None,
        Kd=None,
        gravity_ff=True,
        q_des=None,
        q_des_range=(-np.pi / 4, np.pi / 4),
        q_init=None,
        q_init_range=(-np.pi / 4, np.pi / 4),
        z_normalization=False,
    ):
        # ── Consistency checks ─────────────────────────────────────────
        if same_length and sim_length is None:
            raise ValueError(
                "same_length=True requires sim_length to be set. "
                "Example: sim_length=300."
            )

        if not same_length and T_s is None and Kp is None:
            print(
                "WARNING: same_length=False with randomized T_s produces "
                "variable-length tensors. This is incompatible with "
                "DataLoader batching. Set same_length=True and sim_length, "
                "or fix T_s to a scalar value."
            )

        if same_length and sim_length is not None:
            fastest_T_s = T_s if T_s is not None else T_s_range[0]
            if Kp is not None:
                fastest_T_s = 5.8 / np.sqrt(Kp / m2)
            min_leg_steps = max(2, int(round(leg_time_factor * fastest_T_s / dt)))
            if sim_length < min_leg_steps:
                raise ValueError(
                    f"sim_length={sim_length} is too short for same_length=True: "
                    f"the fastest leg (T_s={fastest_T_s:.2f}s) needs at least "
                    f"{min_leg_steps} steps at dt={dt}. "
                    f"Increase sim_length or reduce leg_time_factor."
                )

        # ── Store config ───────────────────────────────────────────────
        self.dt                    = dt
        self.n_rep                 = n_rep
        self.leg_time_factor       = leg_time_factor
        self.same_length           = same_length
        self.sim_length            = sim_length
        self.angle_limit           = angle_limit
        self.require_convergence   = require_convergence
        self.conv_delta            = conv_delta
        self.max_resample_attempts = max_resample_attempts
        self.l1, self.l2           = l1, l2
        self.m1, self.m2           = m1, m2
        self.g                     = g
        self.gravity_ff            = gravity_ff
        self._Kp_fixed             = Kp
        self._Kd_fixed             = Kd
        self.T_s_fixed             = T_s
        self.T_s_range             = T_s_range
        self._q_des_fixed          = np.array(q_des)  if q_des  is not None else None
        self._q_init_fixed         = np.array(q_init) if q_init is not None else None
        self.q_des_range           = q_des_range
        self.q_init_range          = q_init_range

        self.data                  = []
        self.targets               = []   # q_des per sample
        self.q_inits               = []   # q_init per sample
        self.T_s_log               = []   # T_s used per sample
        self.leg_boundaries        = []   # boundaries in raw trajectory
        self.leg_boundaries_final  = []   # boundaries after length policy
        self.leg_targets           = []   # target angle per leg
        self.convergence_log       = []   # list[bool] per leg, per sample

        # ── Simulate ───────────────────────────────────────────────────
        for _ in range(n_samples):
            traj, q0, q_des_i, T_s_i, raw_bounds, leg_tgts, conv_log = \
                self._simulate_with_guard()

            self.targets.append(q_des_i)
            self.q_inits.append(q0)
            self.T_s_log.append(T_s_i)
            self.leg_boundaries.append(raw_bounds)
            self.leg_targets.append(leg_tgts)
            self.convergence_log.append(conv_log)

            final_bounds = self._apply_length_policy_bounds(
                raw_bounds, len(traj)
            )
            traj = self._apply_length_policy(traj)

            self.leg_boundaries_final.append(final_bounds)
            self.data.append(torch.tensor(traj, dtype=torch.float32))

        self.data_original = self.data.copy()

        if z_normalization:
            all_data = torch.cat(self.data, dim=0)
            self.mean = all_data.mean(dim=0)
            self.std  = all_data.std(dim=0)
            self.data = [(x - self.mean) / self.std for x in self.data]

    # ------------------------------------------------------------------
    # Convergence check (shared by dataset and checker)
    # ------------------------------------------------------------------

    def _check_leg_convergence(
        self, traj_raw: np.ndarray, raw_bounds: list, leg_tgts: list
    ) -> list:
        """
        Evaluate convergence for every leg on the RAW trajectory.

        For each leg, checks whether all joints are within conv_delta
        of the leg's target at the leg's final step.

        Returns a list of bools, one per leg.
        """
        conv = []
        for end_step, target in zip(raw_bounds, leg_tgts):
            end_step = min(end_step, traj_raw.shape[0] - 1)
            error    = np.abs(traj_raw[end_step] - target)
            conv.append(bool(np.all(error <= self.conv_delta)))
        return conv

    # ------------------------------------------------------------------
    # Stability + convergence guard
    # ------------------------------------------------------------------

    def _simulate_with_guard(self):
        """
        Simulate one sample with up to max_resample_attempts tries.

        Stability (angle_limit) is always enforced.
        Convergence (conv_delta) is enforced only when
        require_convergence=True.

        Returns
        -------
        traj, q0, q_des_i, T_s_i, raw_bounds, leg_tgts, conv_log
        """
        last = None

        for attempt in range(self.max_resample_attempts):
            q0, q_des_i, Kp_i, Kd_i, T_s_i = self._sample_params()
            traj, raw_bounds, leg_tgts, stable = \
                self._simulate_trajectory(q0, q_des_i, Kp_i, Kd_i, T_s_i)

            conv_log = self._check_leg_convergence(traj, raw_bounds, leg_tgts)
            all_converged = all(conv_log)

            # store as fallback in case no attempt fully succeeds
            last = (traj, q0, q_des_i, T_s_i, raw_bounds, leg_tgts, conv_log)

            if not stable:
                continue   # always retry on instability

            if self.require_convergence and not all_converged:
                continue   # retry when convergence is required

            return traj, q0, q_des_i, T_s_i, raw_bounds, leg_tgts, conv_log

        # all attempts exhausted
        reason = "instability" if not stable else "non-convergence"
        print(
            f"WARNING: could not find a {'stable and converging' if self.require_convergence else 'stable'} "
            f"trajectory after {self.max_resample_attempts} attempts "
            f"(last failure: {reason}). Keeping last attempt. "
            f"Consider adjusting angle_limit, conv_delta, "
            f"q_init_range/q_des_range, or Kp."
        )
        return last

    # ------------------------------------------------------------------
    # Length policy
    # ------------------------------------------------------------------

    def _apply_length_policy(self, traj: np.ndarray) -> np.ndarray:
        """Resample or trim trajectory according to same_length/sim_length."""
        if self.same_length:
            indices = np.round(
                np.linspace(0, len(traj) - 1, self.sim_length)
            ).astype(int)
            return traj[indices]
        if self.sim_length is not None:
            return traj[: self.sim_length]
        return traj

    def _apply_length_policy_bounds(
        self, raw_bounds: list, raw_len: int
    ) -> list:
        """
        Rescale raw leg boundary indices to match the stored trajectory
        after the length policy has been applied.
        """
        if self.same_length:
            scale = (self.sim_length - 1) / max(raw_len - 1, 1)
            return [min(int(round(b * scale)), self.sim_length - 1)
                    for b in raw_bounds]
        if self.sim_length is not None:
            return [min(b, self.sim_length - 1) for b in raw_bounds]
        return list(raw_bounds)

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self):
        """Sample q_init, q_des, Kp, Kd, T_s for one trajectory."""
        q0 = (
            self._q_init_fixed.copy()
            if self._q_init_fixed is not None
            else np.random.uniform(*self.q_init_range, size=2)
        )
        q_des_i = (
            self._q_des_fixed.copy()
            if self._q_des_fixed is not None
            else np.random.uniform(*self.q_des_range, size=2)
        )

        if self._Kp_fixed is not None:
            Kp_i  = self._Kp_fixed
            Kd_i  = (
                self._Kd_fixed if self._Kd_fixed is not None
                else 2.0 * np.sqrt(Kp_i * self.m2)
            )
            T_s_i = 5.8 / np.sqrt(Kp_i / self.m2)
        else:
            T_s_i = (
                self.T_s_fixed if self.T_s_fixed is not None
                else np.random.uniform(*self.T_s_range)
            )
            Kp_i, Kd_i = self._gains_from_Ts(T_s_i, self.m2)

        return q0, q_des_i, Kp_i, Kd_i, T_s_i

    @staticmethod
    def _gains_from_Ts(T_s, m2):
        """Compute critically-damped PD gains for a desired settling time."""
        omega_n = 5.8 / T_s
        Kp = m2 * omega_n ** 2
        Kd = 2.0 * np.sqrt(m2 * Kp)
        return Kp, Kd

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_leg(self, init_state, q_des_i, Kp_i, Kd_i, T_s_i):
        """
        Simulate one motion leg.
        Returns positions [steps, 2], final full state [4], and a bool
        indicating whether the leg stayed within angle_limit.
        """
        duration = self.leg_time_factor * T_s_i
        n_steps  = max(2, int(round(duration / self.dt)))
        time     = np.arange(n_steps) * self.dt

        sol = odeint(
            self.equations,
            init_state,
            time,
            args=(self.m1, self.m2, self.l1, self.l2,
                  self.g, Kp_i, Kd_i, q_des_i, self.gravity_ff),
        )
        stable = bool(np.all(np.abs(sol[:, :2]) <= self.angle_limit))
        return sol[:, :2], sol[-1], stable

    def _simulate_trajectory(self, q0, q_des_i, Kp_i, Kd_i, T_s_i):
        """
        Simulate the full multi-rep trajectory.

        Returns
        -------
        traj       : np.ndarray [total_steps, 2]
        raw_bounds : list[int] — last step index of each leg (raw)
        leg_tgts   : list[np.ndarray] — target angle for each leg
        stable     : bool — False if any leg exceeded angle_limit
        """
        segments   = []
        raw_bounds = []
        leg_tgts   = []
        state      = np.concatenate([q0, np.zeros(2)])
        step       = 0
        stable     = True

        for rep in range(self.n_rep):
            pos, state, leg_ok = self._simulate_leg(
                state, q_des_i, Kp_i, Kd_i, T_s_i)
            segments.append(pos)
            step += len(pos)
            raw_bounds.append(step - 1)
            leg_tgts.append(q_des_i.copy())
            stable = stable and leg_ok

            if self.n_rep > 1 and rep < self.n_rep - 1:
                pos, state, leg_ok = self._simulate_leg(
                    state, q0, Kp_i, Kd_i, T_s_i)
                segments.append(pos)
                step += len(pos)
                raw_bounds.append(step - 1)
                leg_tgts.append(q0.copy())
                stable = stable and leg_ok

        traj = np.concatenate(segments, axis=0)
        return traj, raw_bounds, leg_tgts, stable

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]   # autoregressive (input, target)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    @staticmethod
    def equations(X, t, m1, m2, l1, l2, g, Kp, Kd, q_des, gravity_ff=True):
        """
        Full 2-link planar arm dynamics.
        M(q)*ddq + C(q,dq)*dq + G(q) = tau
        tau = Kp*(q_des - q) - Kd*dq  [+ G(q_des)]
        """
        q1, q2, dq1, dq2 = X
        c2 = np.cos(q2)
        s2 = np.sin(q2)

        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
        M12 = m2 * l2**2 + m2 * l1 * l2 * c2
        M22 = m2 * l2**2
        M   = np.array([[M11, M12], [M12, M22]])

        h    = m2 * l1 * l2 * s2
        C_dq = np.array([
            -h * dq2 * (dq1 + dq2),
             h * dq1 * dq1
        ])

        G = np.array([
            (m1 + m2) * g * l1 * np.cos(q1) + m2 * g * l2 * np.cos(q1 + q2),
             m2 * g * l2 * np.cos(q1 + q2)
        ])

        q   = np.array([q1, q2])
        dq  = np.array([dq1, dq2])
        tau = Kp * (q_des - q) - Kd * dq

        if gravity_ff:
            q1d, q2d = q_des
            G_des = np.array([
                (m1 + m2) * g * l1 * np.cos(q1d) + m2 * g * l2 * np.cos(q1d + q2d),
                 m2 * g * l2 * np.cos(q1d + q2d)
            ])
            tau = tau + G_des

        ddq = np.linalg.solve(M, tau - C_dq - G)
        return [dq1, dq2, ddq[0], ddq[1]]
    
    
    
    




class TwoLinkArmDataset_withvel(Dataset):
    """
    Dataset simulating a 2-link planar robot arm under PD control,
    with encoder-like position readout (joint angles only).

    State vector: [q1, q2, dq1, dq2]
      - q1, q2   : joint angles (rad)
      - dq1, dq2 : joint velocities (rad/s)

    Dynamics (standard manipulator equation):
        M(q) * ddq + C(q, dq) * dq + g(q) = tau

    Control law (PD with optional gravity feedforward):
        tau = Kp * (q_des - q) - Kd * dq  [+ G(q_des)]

    The output stored is [q1(t), q2(t)] only, mimicking encoder readout.

    Motion structure
    ----------------
    Each sample is built from n_rep repetitions. One repetition = one
    forward leg (q_init → q_des) plus, if n_rep > 1, one return leg
    (q_des → q_init). Legs are simulated sequentially: the end state of
    each leg is the exact initial condition of the next, so velocity
    continuity is preserved.

    Leg boundaries are stored in self.leg_boundaries[i] (raw, pre-policy)
    and self.leg_boundaries_final[i] (after length policy). The checker
    always reads leg_boundaries_final so it stays in sync with the stored
    data regardless of same_length / sim_length settings.

    Convergence check
    -----------------
    A trajectory is "converged" if, at the end of every leg, all joints
    are within conv_delta (rad) of the leg's target. This is evaluated on
    the RAW simulated trajectory (before any length policy), so the check
    is always physically meaningful.

    Two parameters control what happens when convergence fails:

      require_convergence=False (default)
        All samples are kept regardless of convergence. Use this when
        you want the dataset to contain failures too (e.g. for training
        a network to generalise to real-world imperfection).

      require_convergence=True
        Non-converging samples are discarded and re-simulated with fresh
        random parameters (up to max_resample_attempts times). Use this
        for a "clean" dataset. If all attempts fail a warning is printed
        and the last attempt is kept.

    self.convergence_log[i] is a list of bools (one per leg) recording
    whether each leg converged, regardless of require_convergence.

    Stability guard
    ---------------
    Independently of convergence, if any joint angle exceeds angle_limit
    (rad) the trajectory is considered physically unstable and is always
    re-simulated (up to max_resample_attempts). This guard runs first and
    cannot be disabled.

    Speed control
    -------------
    Gesture speed is controlled via the desired settling time T_s (s):
        omega_n = 5.8 / T_s
        Kp      = m2 * omega_n^2
        Kd      = 2 * sqrt(m2 * Kp)   (critically damped)

    Fixed-length output (same_length)
    ----------------------------------
    When same_length=True every trajectory is resampled to sim_length
    steps via index interpolation after simulation.
    When same_length=False sim_length acts as a hard trim.
    Variable-length output (same_length=False, no sim_length) prints a
    warning because it is incompatible with DataLoader batching.

    Parameters
    ----------
    n_samples             : number of independent trajectories
    dt                    : timestep (s)
    n_rep                 : number of forward legs (repetitions).
                            1  → q_init → q_des
                            >1 → q_init → q_des → q_init → ... → q_des
    T_s                   : fixed settling time (s). If None, randomized
                            per sample from T_s_range.
    T_s_range             : (T_s_min, T_s_max) used when T_s is None.
    leg_time_factor       : each leg runs for leg_time_factor * T_s s.
    same_length           : if True resample to sim_length steps.
    sim_length            : output length in steps. Required when
                            same_length=True; acts as trim otherwise.
    angle_limit           : stability threshold (rad). Default pi.
    require_convergence   : if True, discard and re-simulate samples
                            where any leg fails to reach its target
                            within conv_delta. Default False.
    conv_delta            : convergence threshold (rad) per joint.
                            Used when require_convergence=True and
                            always logged in convergence_log.
                            Default 0.05 rad (~3 deg).
    max_resample_attempts : max re-simulation attempts per sample
                            (applies to both stability and convergence
                            guards). Default 10.
    l1, l2                : link lengths (m)
    m1, m2                : link masses (kg)
    g                     : gravitational acceleration (m/s^2)
    Kp                    : fixed PD proportional gain (overrides T_s).
    Kd                    : fixed PD derivative gain.
    gravity_ff            : if True add gravity feedforward to torque.
    q_des                 : fixed desired angles [q1_des, q2_des] (rad).
                            If None randomized from q_des_range.
    q_des_range           : (low, high) uniform range for q_des (rad).
    q_init                : fixed initial angles [q1_0, q2_0] (rad).
                            If None randomized from q_init_range.
    q_init_range          : (low, high) uniform range for q_init (rad).
    z_normalization       : if True standardize output trajectories.
    """

    def __init__(
        self,
        n_samples=1000,
        dt=0.01,
        n_rep=1,
        T_s=None,
        T_s_range=(0.5, 3.0),
        leg_time_factor=2.0,
        same_length=False,
        sim_length=None,
        angle_limit=np.pi,
        require_convergence=False,
        conv_delta=0.05,
        max_resample_attempts=10,
        l1=1.0,
        l2=0.8,
        m1=1.0,
        m2=0.5,
        g=9.81,
        Kp=None,
        Kd=None,
        gravity_ff=True,
        q_des=None,
        q_des_range=(-np.pi / 4, np.pi / 4),
        q_init=None,
        q_init_range=(-np.pi / 4, np.pi / 4),
        z_normalization=True,
    ):
        # ── Consistency checks ─────────────────────────────────────────
        if same_length and sim_length is None:
            raise ValueError(
                "same_length=True requires sim_length to be set. "
                "Example: sim_length=300."
            )

        if not same_length and T_s is None and Kp is None:
            print(
                "WARNING: same_length=False with randomized T_s produces "
                "variable-length tensors. This is incompatible with "
                "DataLoader batching. Set same_length=True and sim_length, "
                "or fix T_s to a scalar value."
            )

        if same_length and sim_length is not None:
            fastest_T_s = T_s if T_s is not None else T_s_range[0]
            if Kp is not None:
                fastest_T_s = 5.8 / np.sqrt(Kp / m2)
            min_leg_steps = max(2, int(round(leg_time_factor * fastest_T_s / dt)))
            if sim_length < min_leg_steps:
                raise ValueError(
                    f"sim_length={sim_length} is too short for same_length=True: "
                    f"the fastest leg (T_s={fastest_T_s:.2f}s) needs at least "
                    f"{min_leg_steps} steps at dt={dt}. "
                    f"Increase sim_length or reduce leg_time_factor."
                )

        # ── Store config ───────────────────────────────────────────────
        self.dt                    = dt
        self.n_rep                 = n_rep
        self.leg_time_factor       = leg_time_factor
        self.same_length           = same_length
        self.sim_length            = sim_length
        self.angle_limit           = angle_limit
        self.require_convergence   = require_convergence
        self.conv_delta            = conv_delta
        self.max_resample_attempts = max_resample_attempts
        self.l1, self.l2           = l1, l2
        self.m1, self.m2           = m1, m2
        self.g                     = g
        self.gravity_ff            = gravity_ff
        self._Kp_fixed             = Kp
        self._Kd_fixed             = Kd
        self.T_s_fixed             = T_s
        self.T_s_range             = T_s_range
        self._q_des_fixed          = np.array(q_des)  if q_des  is not None else None
        self._q_init_fixed         = np.array(q_init) if q_init is not None else None
        self.q_des_range           = q_des_range
        self.q_init_range          = q_init_range

        self.data                  = []
        self.targets               = []   # q_des per sample
        self.q_inits               = []   # q_init per sample
        self.T_s_log               = []   # T_s used per sample
        self.leg_boundaries        = []   # boundaries in raw trajectory
        self.leg_boundaries_final  = []   # boundaries after length policy
        self.leg_targets           = []   # target angle per leg
        self.convergence_log       = []   # list[bool] per leg, per sample

        # ── Simulate ───────────────────────────────────────────────────
        for _ in range(n_samples):
            traj, q0, q_des_i, T_s_i, raw_bounds, leg_tgts, conv_log = \
                self._simulate_with_guard()

            self.targets.append(q_des_i)
            self.q_inits.append(q0)
            self.T_s_log.append(T_s_i)
            self.leg_boundaries.append(raw_bounds)
            self.leg_targets.append(leg_tgts)
            self.convergence_log.append(conv_log)

            final_bounds = self._apply_length_policy_bounds(
                raw_bounds, len(traj)
            )
            traj = self._apply_length_policy(traj)

            self.leg_boundaries_final.append(final_bounds)
            self.data.append(torch.tensor(traj, dtype=torch.float32))

        self.data_original = self.data.copy()

        if z_normalization:
            all_data = torch.cat(self.data, dim=0)
            self.mean = all_data.mean(dim=0)
            self.std  = all_data.std(dim=0)
            self.data = [(x - self.mean) / self.std for x in self.data]

    # ------------------------------------------------------------------
    # Convergence check (shared by dataset and checker)
    # ------------------------------------------------------------------

    def _check_leg_convergence(
        self, traj_raw: np.ndarray, raw_bounds: list, leg_tgts: list
    ) -> list:
        """
        Evaluate convergence for every leg on the RAW trajectory.

        For each leg, checks whether all joints are within conv_delta
        of the leg's target at the leg's final step.

        Returns a list of bools, one per leg.
        """
        conv = []
        for end_step, target in zip(raw_bounds, leg_tgts):
            end_step = min(end_step, traj_raw.shape[0] - 1)
            error    = np.abs(traj_raw[end_step][:2] - target)
            conv.append(bool(np.all(error <= self.conv_delta)))
        return conv

    # ------------------------------------------------------------------
    # Stability + convergence guard
    # ------------------------------------------------------------------

    def _simulate_with_guard(self):
        """
        Simulate one sample with up to max_resample_attempts tries.

        Stability (angle_limit) is always enforced.
        Convergence (conv_delta) is enforced only when
        require_convergence=True.

        Returns
        -------
        traj, q0, q_des_i, T_s_i, raw_bounds, leg_tgts, conv_log
        """
        last = None

        for attempt in range(self.max_resample_attempts):
            q0, q_des_i, Kp_i, Kd_i, T_s_i = self._sample_params()
            traj, raw_bounds, leg_tgts, stable = \
                self._simulate_trajectory(q0, q_des_i, Kp_i, Kd_i, T_s_i)

            conv_log = self._check_leg_convergence(traj, raw_bounds, leg_tgts)
            all_converged = all(conv_log)

            # store as fallback in case no attempt fully succeeds
            last = (traj, q0, q_des_i, T_s_i, raw_bounds, leg_tgts, conv_log)

            if not stable:
                continue   # always retry on instability

            if self.require_convergence and not all_converged:
                continue   # retry when convergence is required

            return traj, q0, q_des_i, T_s_i, raw_bounds, leg_tgts, conv_log

        # all attempts exhausted
        reason = "instability" if not stable else "non-convergence"
        print(
            f"WARNING: could not find a {'stable and converging' if self.require_convergence else 'stable'} "
            f"trajectory after {self.max_resample_attempts} attempts "
            f"(last failure: {reason}). Keeping last attempt. "
            f"Consider adjusting angle_limit, conv_delta, "
            f"q_init_range/q_des_range, or Kp."
        )
        return last

    # ------------------------------------------------------------------
    # Length policy
    # ------------------------------------------------------------------

    def _apply_length_policy(self, traj: np.ndarray) -> np.ndarray:
        """Resample or trim trajectory according to same_length/sim_length."""
        if self.same_length:
            indices = np.round(
                np.linspace(0, len(traj) - 1, self.sim_length)
            ).astype(int)
            return traj[indices]
        if self.sim_length is not None:
            return traj[: self.sim_length]
        return traj

    def _apply_length_policy_bounds(
        self, raw_bounds: list, raw_len: int
    ) -> list:
        """
        Rescale raw leg boundary indices to match the stored trajectory
        after the length policy has been applied.
        """
        if self.same_length:
            scale = (self.sim_length - 1) / max(raw_len - 1, 1)
            return [min(int(round(b * scale)), self.sim_length - 1)
                    for b in raw_bounds]
        if self.sim_length is not None:
            return [min(b, self.sim_length - 1) for b in raw_bounds]
        return list(raw_bounds)

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def _sample_params(self):
        """Sample q_init, q_des, Kp, Kd, T_s for one trajectory."""
        q0 = (
            self._q_init_fixed.copy()
            if self._q_init_fixed is not None
            else np.random.uniform(*self.q_init_range, size=2)
        )
        q_des_i = (
            self._q_des_fixed.copy()
            if self._q_des_fixed is not None
            else np.random.uniform(*self.q_des_range, size=2)
        )

        if self._Kp_fixed is not None:
            Kp_i  = self._Kp_fixed
            Kd_i  = (
                self._Kd_fixed if self._Kd_fixed is not None
                else 2.0 * np.sqrt(Kp_i * self.m2)
            )
            T_s_i = 5.8 / np.sqrt(Kp_i / self.m2)
        else:
            T_s_i = (
                self.T_s_fixed if self.T_s_fixed is not None
                else np.random.uniform(*self.T_s_range)
            )
            Kp_i, Kd_i = self._gains_from_Ts(T_s_i, self.m2)

        return q0, q_des_i, Kp_i, Kd_i, T_s_i

    @staticmethod
    def _gains_from_Ts(T_s, m2):
        """Compute critically-damped PD gains for a desired settling time."""
        omega_n = 5.8 / T_s
        Kp = m2 * omega_n ** 2
        Kd = 2.0 * np.sqrt(m2 * Kp)
        return Kp, Kd

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_leg(self, init_state, q_des_i, Kp_i, Kd_i, T_s_i):
        """
        Simulate one motion leg.
        Returns positions and velocities [steps, 4], final full state [4], and a bool
        indicating whether the leg stayed within angle_limit.
        """
        duration = self.leg_time_factor * T_s_i
        n_steps  = max(2, int(round(duration / self.dt)))
        time     = np.arange(n_steps) * self.dt

        sol = odeint(
            self.equations,
            init_state,
            time,
            args=(self.m1, self.m2, self.l1, self.l2,
                  self.g, Kp_i, Kd_i, q_des_i, self.gravity_ff),
        )
        stable = bool(np.all(np.abs(sol[:, :2]) <= self.angle_limit))
        return sol[:, :4], sol[-1], stable

    def _simulate_trajectory(self, q0, q_des_i, Kp_i, Kd_i, T_s_i):
        """
        Simulate the full multi-rep trajectory.

        Returns
        -------
        traj       : np.ndarray [total_steps, 2]
        raw_bounds : list[int] — last step index of each leg (raw)
        leg_tgts   : list[np.ndarray] — target angle for each leg
        stable     : bool — False if any leg exceeded angle_limit
        """
        segments   = []
        raw_bounds = []
        leg_tgts   = []
        state      = np.concatenate([q0, np.zeros(2)])
        step       = 0
        stable     = True

        for rep in range(self.n_rep):
            pos, state, leg_ok = self._simulate_leg(
                state, q_des_i, Kp_i, Kd_i, T_s_i)
            segments.append(pos)
            step += len(pos)
            raw_bounds.append(step - 1)
            leg_tgts.append(q_des_i.copy())
            stable = stable and leg_ok

            if self.n_rep > 1 and rep < self.n_rep - 1:
                pos, state, leg_ok = self._simulate_leg(
                    state, q0, Kp_i, Kd_i, T_s_i)
                segments.append(pos)
                step += len(pos)
                raw_bounds.append(step - 1)
                leg_tgts.append(q0.copy())
                stable = stable and leg_ok

        traj = np.concatenate(segments, axis=0)
        return traj, raw_bounds, leg_tgts, stable

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]   # autoregressive (input, target)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    @staticmethod
    def equations(X, t, m1, m2, l1, l2, g, Kp, Kd, q_des, gravity_ff=True):
        """
        Full 2-link planar arm dynamics.
        M(q)*ddq + C(q,dq)*dq + G(q) = tau
        tau = Kp*(q_des - q) - Kd*dq  [+ G(q_des)]
        """
        q1, q2, dq1, dq2 = X
        c2 = np.cos(q2)
        s2 = np.sin(q2)

        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
        M12 = m2 * l2**2 + m2 * l1 * l2 * c2
        M22 = m2 * l2**2
        M   = np.array([[M11, M12], [M12, M22]])

        h    = m2 * l1 * l2 * s2
        C_dq = np.array([
            -h * dq2 * (dq1 + dq2),
             h * dq1 * dq1
        ])

        G = np.array([
            (m1 + m2) * g * l1 * np.cos(q1) + m2 * g * l2 * np.cos(q1 + q2),
             m2 * g * l2 * np.cos(q1 + q2)
        ])

        q   = np.array([q1, q2])
        dq  = np.array([dq1, dq2])
        tau = Kp * (q_des - q) - Kd * dq

        if gravity_ff:
            q1d, q2d = q_des
            G_des = np.array([
                (m1 + m2) * g * l1 * np.cos(q1d) + m2 * g * l2 * np.cos(q1d + q2d),
                 m2 * g * l2 * np.cos(q1d + q2d)
            ])
            tau = tau + G_des

        ddq = np.linalg.solve(M, tau - C_dq - G)
        return [dq1, dq2, ddq[0], ddq[1]]