import torch
import torch.nn as nn

class DynamicsNet(nn.Module):
    """
    Small MLP that learns the implicit dynamics of the system.
    Given the current state (q, q_dot), predicts the time derivative (dq/dt, dq_dot/dt).
    
    This makes NO assumptions about the system - it learns purely from the
    SNN's predictions, acting as a learned consistency constraint.
    """
    
    def __init__(self, state_dim, hidden_dim=64, dt=1.0, device="cuda", dtype=torch.float):
        """
        Args:
            state_dim (int): Dimensionality of q (and q_dot). Total input is 2*state_dim.
            hidden_dim (int): Width of hidden layers.
            dt (float): Time step size.
            device (str): Device.
            dtype (torch.dtype): Dtype.
        """
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt
        self.device = device
        self.dtype = dtype
        
        # Input: concatenation of [q, q_dot] -> 2*state_dim
        # Output: [dq/dt, dq_dot/dt] -> 2*state_dim
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * state_dim)
        ).to(device=device, dtype=dtype)
        
        # Separate optimizer - trained jointly but independently from the SNN
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, q, q_dot):
        """
        Args:
            q     (torch.Tensor): Positions,  shape (B, state_dim)
            q_dot (torch.Tensor): Velocities, shape (B, state_dim)
        Returns:
            dq_dt     (torch.Tensor): Predicted dq/dt,     shape (B, state_dim)
            dq_dot_dt (torch.Tensor): Predicted dq_dot/dt, shape (B, state_dim)
        """
        x = torch.cat([q, q_dot], dim=-1)   # (B, 2*state_dim)
        out = self.net(x)                    # (B, 2*state_dim)
        dq_dt     = out[..., :self.state_dim]
        dq_dot_dt = out[..., self.state_dim:]
        return dq_dt, dq_dot_dt
    
    def consistency_loss(self, q_seq, q_dot_seq):
        """
        Computes the dynamics consistency loss over a predicted trajectory.
        
        Penalizes the SNN predictions for violating the learned dynamics:
        
            || (q_{t+1} - q_t) / dt    - f_phi^q(q_t, q_dot_t)    ||^2
          + || (q_dot_{t+1} - q_dot_t) / dt - f_phi^qdot(q_t, q_dot_t) ||^2
        
        Args:
            q_seq     (torch.Tensor): Predicted positions,  shape (B, T, state_dim)
            q_dot_seq (torch.Tensor): Predicted velocities, shape (B, T, state_dim)
        Returns:
            loss (torch.Tensor): Scalar consistency loss.
        """
        # Finite difference approximation of derivatives from predictions
        # shape: (B, T-1, state_dim)
        dq_fd     = (q_seq[:, 1:, :]     - q_seq[:, :-1, :])     / self.dt
        dq_dot_fd = (q_dot_seq[:, 1:, :] - q_dot_seq[:, :-1, :]) / self.dt
        
        # Predicted derivatives from dynamics net at each t
        # We evaluate f_phi at all t except the last
        B, T, D = q_seq.shape
        q_flat     = q_seq[:, :-1, :].reshape(B * (T-1), D)
        q_dot_flat = q_dot_seq[:, :-1, :].reshape(B * (T-1), D)
        
        dq_pred, dq_dot_pred = self.forward(q_flat, q_dot_flat)
        dq_pred     = dq_pred.reshape(B, T-1, D)
        dq_dot_pred = dq_dot_pred.reshape(B, T-1, D)
        
        loss = (
            torch.mean((dq_fd - dq_pred) ** 2) +
            torch.mean((dq_dot_fd - dq_dot_pred) ** 2)
        )
        return loss
    
    def self_supervised_update(self, q_seq, q_dot_seq):
        """
        Updates f_phi using the SNN's own predictions as supervision.
        This is called BEFORE the SNN update so f_phi is always one step ahead.
        
        f_phi learns: given (q_t, q_dot_t), predict (q_{t+1}-q_t)/dt
        using the SNN predictions as the ground truth for this update.
        
        Args:
            q_seq     (torch.Tensor): Predicted positions,  shape (B, T, state_dim)
            q_dot_seq (torch.Tensor): Predicted velocities, shape (B, T, state_dim)
        Returns:
            f_loss (torch.Tensor): Scalar loss for f_phi's self-supervised update.
        """
        # Detach from SNN graph - f_phi trains on SNN output as fixed targets
        q_seq     = q_seq.detach()
        q_dot_seq = q_dot_seq.detach()
        
        dq_fd     = (q_seq[:, 1:, :]     - q_seq[:, :-1, :])     / self.dt
        dq_dot_fd = (q_dot_seq[:, 1:, :] - q_dot_seq[:, :-1, :]) / self.dt
        
        B, T, D = q_seq.shape
        q_flat     = q_seq[:, :-1, :].reshape(B * (T-1), D)
        q_dot_flat = q_dot_seq[:, :-1, :].reshape(B * (T-1), D)
        
        dq_pred, dq_dot_pred = self.forward(q_flat, q_dot_flat)
        dq_pred     = dq_pred.reshape(B, T-1, D)
        dq_dot_pred = dq_dot_pred.reshape(B, T-1, D)
        
        f_loss = (
            torch.mean((dq_fd - dq_pred) ** 2) +
            torch.mean((dq_dot_fd - dq_dot_pred) ** 2)
        )
        
        self.optimizer.zero_grad()
        f_loss.backward()
        self.optimizer.step()
        
        return f_loss.item()
    
    
    
class PhysicsConsistentLoss(nn.Module):
    """
    Combined loss: MSE + dynamics consistency.
    
    L = L_MSE + lambda_c * L_consistency + lambda_fd * L_finite_diff
    
    L_MSE         : standard supervised loss on (q, q_dot)
    L_consistency : SNN predictions must agree with f_phi's learned dynamics
    L_finite_diff : cheap kinematic constraint: q_dot ≈ (q_{t+1} - q_t) / dt
    """
    
    def __init__(self, dynamics_net, state_dim, lambda_c=0.1, lambda_fd=0.05, dt=1.0):
        """
        Args:
            dynamics_net (DynamicsNet): The jointly trained f_phi.
            state_dim    (int)        : Dimensionality of q (= dimensionality of q_dot).
            lambda_c     (float)      : Weight for dynamics consistency loss.
            lambda_fd    (float)      : Weight for finite difference consistency loss.
            dt           (float)      : Time step.
        """
        super().__init__()
        self.dynamics_net = dynamics_net
        self.state_dim    = state_dim
        self.lambda_c     = lambda_c
        self.lambda_fd    = lambda_fd
        self.dt           = dt
    
    def forward(self, pred, target):
        """
        Args:
            pred   (torch.Tensor): SNN predictions, shape (B, T, 2*state_dim)
                                   assumed order: [q | q_dot] along last dim
            target (torch.Tensor): Ground truth,   shape (B, T, 2*state_dim)
        Returns:
            total_loss  (torch.Tensor): Scalar.
            loss_dict   (dict)        : Individual loss components for logging.
        """
        # --- Split into positions and velocities ---
        q_pred     = pred[...,   :self.state_dim]   # (B, T, state_dim)
        q_dot_pred = pred[...,   self.state_dim:]   # (B, T, state_dim)
        q_target   = target[..., :self.state_dim]
        q_dot_target = target[..., self.state_dim:]
        
        # --- 1. Standard MSE ---
        loss_mse = (
            torch.mean((q_pred - q_target) ** 2) +
            torch.mean((q_dot_pred - q_dot_target) ** 2)
        )
        
        # --- 2. Finite difference consistency (cheap, no f_phi needed) ---
        # Enforces: q_dot_t ≈ (q_{t+1} - q_t) / dt in the PREDICTIONS
        dq_fd = (q_pred[:, 1:, :] - q_pred[:, :-1, :]) / self.dt
        loss_fd = torch.mean((q_dot_pred[:, :-1, :] - dq_fd) ** 2)
        
        # --- 3. Dynamics consistency via f_phi ---
        loss_c = self.dynamics_net.consistency_loss(q_pred, q_dot_pred)
        
        # --- Total ---
        total_loss = loss_mse + self.lambda_fd * loss_fd + self.lambda_c * loss_c
        
        loss_dict = {
            "mse":         loss_mse.item(),
            "finite_diff": loss_fd.item(),
            "consistency": loss_c.item(),
            "total":       total_loss.item()
        }
        
        return total_loss, loss_dict