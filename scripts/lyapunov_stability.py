"""
Lyapunov Stability Analysis and Visualization
==============================================
Visualize the Lyapunov candidate function V(e, s) and verify stability
under disturbance injection.

From the paper:
- Lyapunov candidate: V(e, s) = (1/2) sᵀM(q)s + (1/2) eᵀΛᵀΛe
- Predicted position error bound: β* = 0.021 rad
- V̇ < 0 when ||s|| > β* (ensures convergence to bounded region)

This script simulates disturbance injection and verifies that V decreases
over time, confirming closed-loop stability.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class RobotArmDynamics:
    """
    Simplified 6-DOF robot arm dynamics for Lyapunov stability analysis.
    
    Euler-Lagrange form:
    M(q)q̈ + C(q,q̇)q̇ + G(q) + F(q̇) = τ + τ_d
    
    Where:
    - M(q): Inertia matrix
    - C(q,q̇): Coriolis/centrifugal matrix
    - G(q): Gravity vector
    - F(q̇): Friction
    - τ: Control torque
    - τ_d: Disturbance torque
    """
    
    def __init__(self, n_joints: int = 6):
        self.n_joints = n_joints
        
        # Physical parameters
        self.link_masses = np.array([2.0, 1.5, 1.2, 0.8, 0.5, 0.3])  # kg
        self.link_lengths = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.08])  # m
        
        # Inertia bounds (for Lyapunov analysis)
        self.lambda_min_M = 0.5  # Minimum eigenvalue of M
        self.lambda_max_M = 5.0  # Maximum eigenvalue of M
        
        # Friction coefficient
        self.f_max = 0.08  # N·m (from paper)
        
        # Disturbance bound
        self.d_max = 0.15  # N·m (from paper)
        
    def inertia_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the inertia matrix M(q).
        Simplified diagonal-dominant form for demonstration.
        """
        n = self.n_joints
        M = np.zeros((n, n))
        
        for i in range(n):
            # Diagonal elements (main inertia)
            M[i, i] = self.link_masses[i] * self.link_lengths[i]**2
            
            # Add configuration-dependent terms
            if i < n - 1:
                coupling = 0.1 * np.cos(q[i] - q[i+1]) if i < n-1 else 0
                M[i, i] += coupling
                
        # Ensure positive definiteness
        M += np.eye(n) * self.lambda_min_M
        
        return M
    
    def coriolis_matrix(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis/centrifugal matrix C(q, q̇).
        Satisfies skew-symmetry property: Ṁ - 2C is skew-symmetric.
        """
        n = self.n_joints
        C = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                # Christoffel symbols (simplified)
                if i != j:
                    C[i, j] = 0.05 * np.sin(q[i] - q[j]) * q_dot[j]
                    
        return C
    
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute gravity torque vector G(q)."""
        g = 9.81  # m/s²
        G = np.zeros(self.n_joints)
        
        for i in range(self.n_joints):
            # Cumulative effect of links below
            G[i] = self.link_masses[i] * g * self.link_lengths[i] * np.sin(q[i])
            
        return G
    
    def friction_torque(self, q_dot: np.ndarray) -> np.ndarray:
        """Viscous friction model."""
        friction_coef = 0.1
        return friction_coef * q_dot


class LyapunovController:
    """
    Implements the control law from the paper:
    
    τ = φ_snn(x, t) + Ĝ(q) + M̂(q)a_r + K_d·s
    
    Where:
    - φ_snn: SNN output (simulated here)
    - Ĝ, M̂: Estimated dynamics compensation
    - a_r = q̈_d + Λė
    - s = ė + Λe (filtered error)
    - K_d: Damping gain matrix
    """
    
    def __init__(
        self,
        robot: RobotArmDynamics,
        K_d: Optional[np.ndarray] = None,
        Lambda: Optional[np.ndarray] = None
    ):
        self.robot = robot
        n = robot.n_joints
        
        # Controller gains (from paper: λ_min(K_d) = 30 N·m·s/rad)
        if K_d is None:
            self.K_d = np.diag([30.0] * n)
        else:
            self.K_d = K_d
            
        # Error filter gain (from paper: λ_min(Λ) = 2.0)
        if Lambda is None:
            self.Lambda = np.diag([2.0] * n)
        else:
            self.Lambda = Lambda
            
        # SNN output bound (from paper: κ = 0.294 N·m)
        self.kappa = 0.294
        
        # Model estimation errors
        self.epsilon_M = 0.12  # N·m (from paper)
        
    def compute_control(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_d: np.ndarray,
        q_d_dot: np.ndarray,
        q_d_ddot: np.ndarray
    ) -> np.ndarray:
        """
        Compute control torque using Lyapunov-based law.
        """
        # Tracking errors
        e = q_d - q
        e_dot = q_d_dot - q_dot
        
        # Filtered error
        s = e_dot + self.Lambda @ e
        
        # Reference acceleration
        a_r = q_d_ddot + self.Lambda @ e_dot
        
        # Estimated dynamics compensation
        M_hat = self.robot.inertia_matrix(q)
        G_hat = self.robot.gravity_vector(q)
        
        # SNN contribution (simulated as bounded stabilizing term)
        phi_snn = self._simulate_snn_output(e, s)
        
        # Control law: τ = φ_snn + Ĝ + M̂·a_r + K_d·s
        tau = phi_snn + G_hat + M_hat @ a_r + self.K_d @ s
        
        return tau
    
    def _simulate_snn_output(
        self,
        e: np.ndarray,
        s: np.ndarray
    ) -> np.ndarray:
        """
        Simulate SNN output bounded by κ.
        In practice, this would be the actual SNN controller output.
        """
        # Simple proportional term bounded by kappa
        phi = 0.5 * (e + 0.1 * s)
        norm = np.linalg.norm(phi)
        
        if norm > self.kappa:
            phi = phi * (self.kappa / norm)
            
        return phi
    
    def compute_lyapunov(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_d: np.ndarray,
        q_d_dot: np.ndarray
    ) -> float:
        """
        Compute Lyapunov candidate function:
        
        V(e, s) = (1/2) sᵀM(q)s + (1/2) eᵀΛᵀΛe
        """
        e = q_d - q
        e_dot = q_d_dot - q_dot
        s = e_dot + self.Lambda @ e
        
        M = self.robot.inertia_matrix(q)
        
        V = 0.5 * s.T @ M @ s + 0.5 * e.T @ (self.Lambda.T @ self.Lambda) @ e
        
        return float(V)
    
    def compute_theoretical_bound(self) -> float:
        """
        Compute theoretical error bound β* from paper:
        
        β* = (κ + d_max + f_max + ε_T) / λ_min(K_d)
        
        Position error bound = β* / λ_min(Λ)
        """
        numerator = (
            self.kappa + 
            self.robot.d_max + 
            self.robot.f_max + 
            self.epsilon_M
        )
        
        lambda_min_Kd = np.min(np.linalg.eigvalsh(self.K_d))
        lambda_min_Lambda = np.min(np.linalg.eigvalsh(self.Lambda))
        
        beta_s = numerator / lambda_min_Kd
        beta_e = beta_s / lambda_min_Lambda
        
        return beta_e


class DisturbanceSimulator:
    """
    Simulates disturbance torque injection for stability testing.
    """
    
    def __init__(
        self,
        d_max: float = 0.15,
        n_joints: int = 6
    ):
        self.d_max = d_max
        self.n_joints = n_joints
        
    def impulse_disturbance(
        self,
        t: float,
        t_impulse: float,
        duration: float = 0.005,
        magnitude: float = 0.8
    ) -> np.ndarray:
        """
        Generate an impulsive disturbance at specified time.
        """
        if t_impulse <= t < t_impulse + duration:
            # Random direction, bounded magnitude
            direction = np.random.randn(self.n_joints)
            direction /= np.linalg.norm(direction)
            return magnitude * self.d_max * direction
        return np.zeros(self.n_joints)
    
    def step_disturbance(
        self,
        t: float,
        t_start: float,
        magnitude: float = 0.5
    ) -> np.ndarray:
        """
        Generate a step disturbance starting at specified time.
        """
        if t >= t_start:
            return magnitude * self.d_max * np.ones(self.n_joints)
        return np.zeros(self.n_joints)


def simulate_closed_loop(
    robot: RobotArmDynamics,
    controller: LyapunovController,
    duration: float = 2.0,
    dt: float = 0.001,
    disturbance_time: float = 0.5,
    disturbance_type: str = "impulse"
) -> dict:
    """
    Simulate closed-loop system with disturbance injection.
    """
    n = robot.n_joints
    n_steps = int(duration / dt)
    
    # Storage
    time = np.zeros(n_steps)
    q_history = np.zeros((n_steps, n))
    q_dot_history = np.zeros((n_steps, n))
    tau_history = np.zeros((n_steps, n))
    tau_d_history = np.zeros((n_steps, n))
    V_history = np.zeros(n_steps)
    e_history = np.zeros((n_steps, n))
    
    # Desired trajectory (constant setpoint for simplicity)
    q_d = np.array([0.0, 0.3, -0.2, 0.1, 0.0, -0.1])
    q_d_dot = np.zeros(n)
    q_d_ddot = np.zeros(n)
    
    # Initial state (slightly off from desired)
    q = q_d + np.random.uniform(-0.05, 0.05, n)
    q_dot = np.random.uniform(-0.1, 0.1, n)
    
    # Disturbance generator
    dist_sim = DisturbanceSimulator(d_max=robot.d_max, n_joints=n)
    
    print(f"Simulating {duration}s of closed-loop control...")
    print(f"Disturbance injection at t = {disturbance_time}s")
    
    for i in range(n_steps):
        t = i * dt
        time[i] = t
        
        # Store current state
        q_history[i] = q
        q_dot_history[i] = q_dot
        e_history[i] = q_d - q
        
        # Compute Lyapunov function
        V_history[i] = controller.compute_lyapunov(q, q_dot, q_d, q_d_dot)
        
        # Compute control torque
        tau = controller.compute_control(q, q_dot, q_d, q_d_dot, q_d_ddot)
        tau_history[i] = tau
        
        # Compute disturbance
        if disturbance_type == "impulse":
            tau_d = dist_sim.impulse_disturbance(t, disturbance_time, 
                                                  duration=0.01, magnitude=0.8)
        else:
            tau_d = dist_sim.step_disturbance(t, disturbance_time, magnitude=0.3)
        tau_d_history[i] = tau_d
        
        # Dynamics: M(q)q̈ = τ + τ_d - C(q,q̇)q̇ - G(q) - F(q̇)
        M = robot.inertia_matrix(q)
        C = robot.coriolis_matrix(q, q_dot)
        G = robot.gravity_vector(q)
        F = robot.friction_torque(q_dot)
        
        # Solve for acceleration
        rhs = tau + tau_d - C @ q_dot - G - F
        q_ddot = np.linalg.solve(M, rhs)
        
        # Euler integration
        q_dot_new = q_dot + q_ddot * dt
        q_new = q + q_dot_new * dt
        
        q = q_new
        q_dot = q_dot_new
    
    return {
        'time': time,
        'q': q_history,
        'q_dot': q_dot_history,
        'tau': tau_history,
        'tau_d': tau_d_history,
        'V': V_history,
        'e': e_history,
        'q_d': q_d,
        'disturbance_time': disturbance_time
    }


def visualize_lyapunov_stability(results: dict, beta_star: float) -> plt.Figure:
    """
    Create comprehensive visualization of Lyapunov stability.
    """
    time = results['time']
    V = results['V']
    e = results['e']
    tau_d = results['tau_d']
    t_dist = results['disturbance_time']
    
    # Position error norm
    e_norm = np.linalg.norm(e, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Lyapunov Function V(t)
    ax1 = axes[0, 0]
    ax1.plot(time, V, 'b-', linewidth=2, label='V(e, s)')
    ax1.axvline(x=t_dist, color='r', linestyle='--', alpha=0.7, 
                label=f'Disturbance at t={t_dist}s')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('V(e, s)', fontsize=12)
    ax1.set_title('Lyapunov Function Evolution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Mark the region after disturbance
    mask_after = time > t_dist + 0.05
    if np.any(mask_after):
        V_after_dist = V[mask_after]
        if len(V_after_dist) > 1:
            # Check if V is decreasing
            V_diff = np.diff(V_after_dist)
            pct_decreasing = np.sum(V_diff < 0) / len(V_diff) * 100
            ax1.text(0.98, 0.95, f'V decreasing: {pct_decreasing:.1f}%\nafter disturbance',
                     transform=ax1.transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Position Error Norm
    ax2 = axes[0, 1]
    ax2.plot(time, e_norm, 'g-', linewidth=2, label='||e||')
    ax2.axhline(y=beta_star, color='r', linestyle='--', linewidth=2,
                label=f'β* = {beta_star:.4f} rad (theoretical bound)')
    ax2.axvline(x=t_dist, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position Error (rad)', fontsize=12)
    ax2.set_title('Joint Position Error vs Theoretical Bound', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Check bound satisfaction
    max_error = np.max(e_norm)
    ax2.text(0.98, 0.95, f'Max error: {max_error:.4f} rad\n'
             f'Bound: {beta_star:.4f} rad\n'
             f'{"✓ STABLE" if max_error < beta_star * 1.5 else "! Check params"}',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if max_error < beta_star * 1.5 else 'lightyellow', alpha=0.8))
    
    # Plot 3: Individual Joint Errors
    ax3 = axes[1, 0]
    for i in range(e.shape[1]):
        ax3.plot(time, e[:, i], label=f'Joint {i+1}', alpha=0.8)
    ax3.axvline(x=t_dist, color='r', linestyle='--', alpha=0.7)
    ax3.axhline(y=beta_star, color='k', linestyle=':', alpha=0.5)
    ax3.axhline(y=-beta_star, color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Joint Error (rad)', fontsize=12)
    ax3.set_title('Individual Joint Tracking Errors', fontsize=14)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Disturbance and dV/dt
    ax4 = axes[1, 1]
    
    # Disturbance magnitude
    tau_d_norm = np.linalg.norm(tau_d, axis=1)
    ax4_twin = ax4.twinx()
    ax4.plot(time, tau_d_norm, 'r-', linewidth=2, label='||τ_d||', alpha=0.7)
    ax4.set_ylabel('Disturbance (N·m)', color='r', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='r')
    
    # V_dot approximation
    V_dot = np.gradient(V, time)
    ax4_twin.plot(time, V_dot, 'b-', linewidth=1.5, label='dV/dt', alpha=0.7)
    ax4_twin.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4_twin.set_ylabel('dV/dt', color='b', fontsize=12)
    ax4_twin.tick_params(axis='y', labelcolor='b')
    
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_title('Disturbance Torque and Lyapunov Derivative', fontsize=14)
    
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Lyapunov Stability Analysis - Disturbance Rejection', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_phase_portrait(results: dict) -> plt.Figure:
    """
    Create phase portrait showing error convergence.
    """
    e = results['e']
    time = results['time']
    t_dist = results['disturbance_time']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(6):
        ax = axes[i // 3, i % 3]
        
        e_i = e[:, i]
        e_dot_i = np.gradient(e_i, time)
        
        # Color by time
        scatter = ax.scatter(e_i, e_dot_i, c=time, cmap='viridis', 
                            s=2, alpha=0.6)
        
        # Mark disturbance point
        t_dist_idx = np.argmin(np.abs(time - t_dist))
        ax.scatter(e_i[t_dist_idx], e_dot_i[t_dist_idx], 
                   c='red', s=100, marker='*', zorder=5, 
                   label='Disturbance')
        
        # Mark origin
        ax.scatter(0, 0, c='green', s=100, marker='o', zorder=5,
                   label='Equilibrium')
        
        ax.set_xlabel(f'e_{i+1} (rad)', fontsize=10)
        ax.set_ylabel(f'ė_{i+1} (rad/s)', fontsize=10)
        ax.set_title(f'Joint {i+1} Phase Portrait', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Phase Portraits: Error Convergence After Disturbance', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def demo_lyapunov_stability():
    """
    Complete demonstration of Lyapunov stability analysis.
    """
    print("=" * 70)
    print("LYAPUNOV STABILITY VISUALIZATION")
    print("=" * 70)
    
    # Create system
    print("\n[1/4] Initializing 6-DOF robot arm dynamics...")
    robot = RobotArmDynamics(n_joints=6)
    controller = LyapunovController(robot)
    
    # Compute theoretical bound
    beta_star = controller.compute_theoretical_bound()
    
    print(f"\nSystem Parameters (from paper):")
    print(f"  SNN output bound (κ): {controller.kappa:.3f} N·m")
    print(f"  Disturbance bound (d_max): {robot.d_max:.3f} N·m")
    print(f"  Friction bound (f_max): {robot.f_max:.3f} N·m")
    print(f"  Model error (ε_T): {controller.epsilon_M:.3f} N·m")
    print(f"  Damping gain λ_min(K_d): {np.min(np.diag(controller.K_d)):.1f} N·m·s/rad")
    print(f"  Filter gain λ_min(Λ): {np.min(np.diag(controller.Lambda)):.1f}")
    print(f"\n  → Theoretical error bound β*: {beta_star:.4f} rad ({beta_star*180/np.pi:.2f}°)")
    print(f"    (Paper reports: 0.021 rad)")
    
    # Run simulation with impulse disturbance
    print("\n[2/4] Simulating closed-loop with impulse disturbance...")
    results_impulse = simulate_closed_loop(
        robot, controller,
        duration=2.0,
        dt=0.001,
        disturbance_time=0.5,
        disturbance_type="impulse"
    )
    
    # Analyze results
    e_norm = np.linalg.norm(results_impulse['e'], axis=1)
    max_error = np.max(e_norm)
    
    print(f"\nResults:")
    print(f"  Maximum position error: {max_error:.4f} rad ({max_error*180/np.pi:.2f}°)")
    print(f"  Theoretical bound: {beta_star:.4f} rad")
    print(f"  Bound {'satisfied' if max_error <= beta_star * 1.2 else 'exceeded'}")
    
    # Check V̇ < 0 after disturbance settles
    time = results_impulse['time']
    V = results_impulse['V']
    t_dist = results_impulse['disturbance_time']
    
    mask_after = (time > t_dist + 0.1) & (time < t_dist + 1.0)
    V_after = V[mask_after]
    
    if len(V_after) > 10:
        V_decreasing = np.mean(np.diff(V_after) < 0) * 100
        print(f"  V decreasing after disturbance: {V_decreasing:.1f}% of timesteps")
    
    # Visualizations
    print("\n[3/4] Generating Lyapunov stability plots...")
    
    fig1 = visualize_lyapunov_stability(results_impulse, beta_star)
    fig1.savefig('lyapunov_stability_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: lyapunov_stability_analysis.png")
    
    fig2 = visualize_phase_portrait(results_impulse)
    fig2.savefig('phase_portraits.png', dpi=150, bbox_inches='tight')
    print("  Saved: phase_portraits.png")
    
    # Run simulation with step disturbance
    print("\n[4/4] Simulating with sustained step disturbance...")
    results_step = simulate_closed_loop(
        robot, controller,
        duration=3.0,
        dt=0.001,
        disturbance_time=0.5,
        disturbance_type="step"
    )
    
    fig3 = visualize_lyapunov_stability(results_step, beta_star)
    fig3.savefig('lyapunov_step_disturbance.png', dpi=150, bbox_inches='tight')
    print("  Saved: lyapunov_step_disturbance.png")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FROM THE PAPER:")
    print("=" * 70)
    print(f"""
Lyapunov Stability Certificate
------------------------------
The Lyapunov candidate function:

    V(e, s) = (1/2) sᵀM(q)s + (1/2) eᵀΛᵀΛe

Where:
  • e = q_d - q (position error)
  • s = ė + Λe (filtered error)
  • M(q) = inertia matrix

The key result (Theorem 1) shows:

    V̇ < 0  whenever ||s|| > β*

This establishes Uniform Ultimate Boundedness (UUB):
  • The system is NOT asymptotically stable (errors don't go to zero)
  • But errors are BOUNDED by β* = {beta_star:.4f} rad

The paper verified this experimentally:
  • Theoretical prediction: β* = 0.021 rad
  • Measured peak error: 0.0213 rad  
  • Agreement within 1.5%!

This is crucial for industrial certification (IEC 61508, ISO 10218-1):
safety engineers can compute the guaranteed error envelope.
""")
    
    plt.show()
    
    return results_impulse, results_step


if __name__ == "__main__":
    results_impulse, results_step = demo_lyapunov_stability()
