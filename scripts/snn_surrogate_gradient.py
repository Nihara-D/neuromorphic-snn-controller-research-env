"""
Surrogate Gradient Sandbox - SNN Training (NumPy Implementation)
================================================================
Train a spiking neural network to predict robot arm dynamics using
surrogate gradient backpropagation through time (BPTT).

From the paper:
- "Surrogate-gradient backpropagation through time (BPTT) with the ATan
   surrogate function is used to train the SEL-to-HPL feedforward weights"
- The SNN learns to map current state (joint errors, velocities) to 
  predicted torques or next positions.

This pure NumPy implementation demonstrates the core challenge of neuromorphic
learning: spikes are non-differentiable, so we use surrogate gradients to 
enable backpropagation through the spiking nonlinearity.

No PyTorch/snntorch required - educational implementation from scratch.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from dataclasses import dataclass


# ============================================================================
# Surrogate Gradient Functions
# ============================================================================

def heaviside(x: np.ndarray) -> np.ndarray:
    """Hard threshold (Heaviside step function) - forward pass."""
    return (x > 0).astype(float)


def atan_surrogate_gradient(x: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """
    ATan surrogate gradient for backprop through spikes.
    
    From the paper: "ATan surrogate function is used"
    
    d/dx[atan(alpha * x)] = alpha / (1 + (alpha * x)^2)
    """
    return alpha / (1.0 + (alpha * x) ** 2)


def fast_sigmoid_surrogate(x: np.ndarray, slope: float = 25.0) -> np.ndarray:
    """Alternative: Fast sigmoid surrogate gradient."""
    return slope / (1.0 + slope * np.abs(x)) ** 2


# ============================================================================
# Robot Arm Data Generation
# ============================================================================

class RobotArmDataGenerator:
    """
    Generate synthetic 6-DOF robot arm dynamics data for SNN training.
    
    Simulates simplified Euler-Lagrange dynamics:
    M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
    
    We generate (torque, current_state) → next_position pairs.
    """
    
    def __init__(
        self,
        n_joints: int = 6,
        dt: float = 0.001,  # 1kHz control rate
        mass_range: Tuple[float, float] = (0.5, 2.0),
        friction_coef: float = 0.1
    ):
        self.n_joints = n_joints
        self.dt = dt
        self.mass_range = mass_range
        self.friction_coef = friction_coef
        
        # Simplified inertia matrix (diagonal for simplicity)
        self.M = np.diag(np.random.uniform(*mass_range, n_joints))
        self.M_inv = np.linalg.inv(self.M)
        
    def simulate_step(
        self,
        q: np.ndarray,      # Joint positions (rad)
        q_dot: np.ndarray,  # Joint velocities (rad/s)
        tau: np.ndarray     # Applied torques (N·m)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate one time step of robot dynamics."""
        # Simplified dynamics: M*q̈ = τ - friction - gravity
        gravity = 0.5 * np.sin(q)  # Simplified gravity term
        friction = self.friction_coef * q_dot
        
        # Solve for acceleration
        q_ddot = self.M_inv @ (tau - friction - gravity)
        
        # Euler integration
        q_dot_next = q_dot + q_ddot * self.dt
        q_next = q + q_dot_next * self.dt
        
        return q_next, q_dot_next
    
    def generate_trajectory(
        self,
        n_steps: int = 1000,
        torque_scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a random trajectory with varying torques."""
        states = []
        torques = []
        next_positions = []
        
        # Random initial state
        q = np.random.uniform(-np.pi/4, np.pi/4, self.n_joints)
        q_dot = np.random.uniform(-0.5, 0.5, self.n_joints)
        
        for _ in range(n_steps):
            # Random smooth torque input
            tau = torque_scale * np.random.randn(self.n_joints)
            
            # Record current state and torque
            state = np.concatenate([q, q_dot])
            states.append(state)
            torques.append(tau)
            
            # Simulate one step
            q_next, q_dot_next = self.simulate_step(q, q_dot, tau)
            next_positions.append(q_next)
            
            # Update state
            q, q_dot = q_next, q_dot_next
        
        return np.array(states), np.array(torques), np.array(next_positions)
    
    def generate_dataset(
        self,
        n_trajectories: int = 50,
        steps_per_trajectory: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training dataset: (torques, states) → next_positions"""
        all_inputs = []
        all_targets = []
        
        for i in range(n_trajectories):
            states, torques, next_pos = self.generate_trajectory(steps_per_trajectory)
            
            # Input: concatenate torques and current state
            inputs = np.concatenate([torques, states], axis=1)
            all_inputs.append(inputs)
            all_targets.append(next_pos)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated trajectory {i+1}/{n_trajectories}")
        
        return np.vstack(all_inputs), np.vstack(all_targets)


# ============================================================================
# Leaky Integrate-and-Fire (LIF) Neuron Model
# ============================================================================

@dataclass
class LIFParameters:
    """Parameters for Leaky Integrate-and-Fire neurons."""
    beta: float = 0.9          # Membrane decay rate (leak)
    threshold: float = 1.0     # Spike threshold
    reset: float = 0.0         # Reset potential after spike
    surrogate_alpha: float = 2.0  # ATan surrogate steepness


class LIFLayer:
    """
    Leaky Integrate-and-Fire neuron layer with surrogate gradient support.
    
    Membrane dynamics: V[t] = beta * V[t-1] + I[t]
    Spike: S[t] = Heaviside(V[t] - threshold)
    Reset: V[t] = V[t] * (1 - S[t]) + reset * S[t]
    
    For backprop, we use ATan surrogate gradient at the spike function.
    """
    
    def __init__(self, size: int, params: LIFParameters = None):
        self.size = size
        self.params = params or LIFParameters()
        
        # State variables (will be set during forward)
        self.membrane = None
        self.spikes = None
        
        # Cache for backward pass
        self.pre_spike_membrane = None  # V before thresholding
        
    def init_state(self, batch_size: int):
        """Initialize membrane potentials to zero."""
        self.membrane = np.zeros((batch_size, self.size))
        
    def forward(self, current: np.ndarray) -> np.ndarray:
        """
        Forward pass through LIF neurons.
        
        Args:
            current: Input current (batch_size, size)
            
        Returns:
            spikes: Binary spike output (batch_size, size)
        """
        # Integrate: V = beta * V + I
        self.membrane = self.params.beta * self.membrane + current
        
        # Cache pre-spike membrane for gradient computation
        self.pre_spike_membrane = self.membrane.copy()
        
        # Spike: S = H(V - threshold)
        self.spikes = heaviside(self.membrane - self.params.threshold)
        
        # Reset: V = V * (1 - S)  (soft reset)
        self.membrane = self.membrane * (1.0 - self.spikes)
        
        return self.spikes
    
    def backward(self, grad_spikes: np.ndarray) -> np.ndarray:
        """
        Backward pass using surrogate gradient.
        
        Args:
            grad_spikes: Gradient w.r.t. spikes (batch_size, size)
            
        Returns:
            grad_current: Gradient w.r.t. input current
        """
        # Surrogate gradient: dS/dV ≈ atan_surrogate(V - threshold)
        v_centered = self.pre_spike_membrane - self.params.threshold
        surrogate_grad = atan_surrogate_gradient(v_centered, self.params.surrogate_alpha)
        
        # Chain rule: dL/dI = dL/dS * dS/dV * dV/dI
        # Since dV/dI = 1 at current timestep:
        grad_current = grad_spikes * surrogate_grad
        
        return grad_current


# ============================================================================
# Two-Layer Spiking Neural Network
# ============================================================================

class SpikingNeuralNetwork:
    """
    Two-layer SNN for robot dynamics prediction.
    
    Architecture mirrors the paper's structure:
    - Input → FC → LIF1 → FC → LIF2 → FC → Output
    
    Uses surrogate gradient for BPTT training.
    """
    
    def __init__(
        self,
        input_size: int = 18,     # 6 torques + 6 pos + 6 vel
        hidden_size: int = 64,
        output_size: int = 6,
        n_steps: int = 20,        # Temporal simulation steps
        lif_params: LIFParameters = None
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_steps = n_steps
        
        # Initialize weights with Xavier/Glorot initialization
        scale1 = np.sqrt(2.0 / (input_size + hidden_size))
        scale2 = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale3 = np.sqrt(2.0 / (hidden_size + output_size))
        
        self.W1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        
        self.W2 = np.random.randn(hidden_size, hidden_size) * scale2
        self.b2 = np.zeros(hidden_size)
        
        self.W_out = np.random.randn(hidden_size, output_size) * scale3
        self.b_out = np.zeros(output_size)
        
        # LIF layers
        self.lif1 = LIFLayer(hidden_size, lif_params)
        self.lif2 = LIFLayer(hidden_size, lif_params)
        
        # Gradient accumulators
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None
        self.grad_W_out = None
        self.grad_b_out = None
        
        # Cache for backward pass
        self.cache = {}
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through the SNN.
        
        Args:
            x: Input (batch_size, input_size)
            
        Returns:
            output: Predicted next positions (batch_size, output_size)
            recordings: Dict of spike/membrane recordings
        """
        batch_size = x.shape[0]
        
        # Initialize LIF states
        self.lif1.init_state(batch_size)
        self.lif2.init_state(batch_size)
        
        # Recording arrays
        spk1_rec = []
        spk2_rec = []
        mem1_rec = []
        mem2_rec = []
        
        # Cache for BPTT
        self.cache['input'] = x
        self.cache['h1_pre'] = []
        self.cache['spk1'] = []
        self.cache['h2_pre'] = []
        self.cache['spk2'] = []
        
        # Temporal simulation
        for t in range(self.n_steps):
            # Layer 1: Input → Hidden
            h1_pre = x @ self.W1 + self.b1
            self.cache['h1_pre'].append(h1_pre)
            
            spk1 = self.lif1.forward(h1_pre)
            self.cache['spk1'].append(spk1.copy())
            spk1_rec.append(spk1)
            mem1_rec.append(self.lif1.membrane.copy())
            
            # Layer 2: Hidden → Hidden
            h2_pre = spk1 @ self.W2 + self.b2
            self.cache['h2_pre'].append(h2_pre)
            
            spk2 = self.lif2.forward(h2_pre)
            self.cache['spk2'].append(spk2.copy())
            spk2_rec.append(spk2)
            mem2_rec.append(self.lif2.membrane.copy())
        
        # Output: Sum spikes and decode
        spike_sum = np.sum(spk2_rec, axis=0) / self.n_steps
        self.cache['spike_sum'] = spike_sum
        
        output = spike_sum @ self.W_out + self.b_out
        
        recordings = {
            'spk1': np.array(spk1_rec),  # (n_steps, batch, hidden)
            'spk2': np.array(spk2_rec),
            'mem1': np.array(mem1_rec),
            'mem2': np.array(mem2_rec)
        }
        
        return output, recordings
    
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass with surrogate gradient BPTT.
        
        Args:
            grad_output: Gradient of loss w.r.t. output (batch_size, output_size)
        """
        batch_size = grad_output.shape[0]
        
        # Initialize gradient accumulators
        self.grad_W1 = np.zeros_like(self.W1)
        self.grad_b1 = np.zeros_like(self.b1)
        self.grad_W2 = np.zeros_like(self.W2)
        self.grad_b2 = np.zeros_like(self.b2)
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)
        
        # Output layer gradients
        spike_sum = self.cache['spike_sum']
        self.grad_W_out = spike_sum.T @ grad_output
        self.grad_b_out = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t. spike sum
        grad_spike_sum = grad_output @ self.W_out.T / self.n_steps
        
        # BPTT through time
        grad_membrane2 = np.zeros((batch_size, self.hidden_size))
        grad_membrane1 = np.zeros((batch_size, self.hidden_size))
        
        for t in reversed(range(self.n_steps)):
            # Restore LIF state for this timestep
            self.lif2.pre_spike_membrane = self.cache['h2_pre'][t]
            self.lif1.pre_spike_membrane = self.cache['h1_pre'][t]
            
            # Gradient through LIF2
            grad_spk2 = grad_spike_sum + grad_membrane2 * self.lif2.params.beta
            grad_h2 = self.lif2.backward(grad_spk2)
            
            # Accumulate W2, b2 gradients
            spk1 = self.cache['spk1'][t]
            self.grad_W2 += spk1.T @ grad_h2
            self.grad_b2 += np.sum(grad_h2, axis=0)
            
            # Gradient through spk1
            grad_spk1 = grad_h2 @ self.W2.T + grad_membrane1 * self.lif1.params.beta
            grad_h1 = self.lif1.backward(grad_spk1)
            
            # Accumulate W1, b1 gradients
            x = self.cache['input']
            self.grad_W1 += x.T @ grad_h1
            self.grad_b1 += np.sum(grad_h1, axis=0)
        
        # Average gradients over batch
        self.grad_W1 /= batch_size
        self.grad_b1 /= batch_size
        self.grad_W2 /= batch_size
        self.grad_b2 /= batch_size
        self.grad_W_out /= batch_size
        self.grad_b_out /= batch_size
    
    def update_weights(self, learning_rate: float, weight_decay: float = 1e-4):
        """Update weights using gradient descent with L2 regularization."""
        # L2 regularization gradients
        self.grad_W1 += weight_decay * self.W1
        self.grad_W2 += weight_decay * self.W2
        self.grad_W_out += weight_decay * self.W_out
        
        # Gradient descent update
        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1
        self.W2 -= learning_rate * self.grad_W2
        self.b2 -= learning_rate * self.grad_b2
        self.W_out -= learning_rate * self.grad_W_out
        self.b_out -= learning_rate * self.grad_b_out


# ============================================================================
# Training Loop
# ============================================================================

class SNNTrainer:
    """Training manager for the SNN."""
    
    def __init__(
        self,
        model: SpikingNeuralNetwork,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_losses = []
        self.val_losses = []
        
    def mse_loss(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute MSE loss and gradient."""
        diff = pred - target
        loss = np.mean(diff ** 2)
        grad = 2 * diff / diff.size
        return loss, grad
    
    def train_epoch(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """Train for one epoch."""
        n_samples = len(train_inputs)
        indices = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            x_batch = train_inputs[batch_idx]
            y_batch = train_targets[batch_idx]
            
            # Forward
            pred, _ = self.model.forward(x_batch)
            
            # Loss
            loss, grad = self.mse_loss(pred, y_batch)
            total_loss += loss
            n_batches += 1
            
            # Backward (surrogate gradient BPTT)
            self.model.backward(grad)
            
            # Update
            self.model.update_weights(self.learning_rate, self.weight_decay)
        
        return total_loss / n_batches
    
    def validate(
        self,
        val_inputs: np.ndarray,
        val_targets: np.ndarray,
        batch_size: int = 64
    ) -> float:
        """Evaluate on validation set."""
        n_samples = len(val_inputs)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            x_batch = val_inputs[i:i + batch_size]
            y_batch = val_targets[i:i + batch_size]
            
            pred, _ = self.model.forward(x_batch)
            loss, _ = self.mse_loss(pred, y_batch)
            total_loss += loss
            n_batches += 1
        
        return total_loss / n_batches
    
    def train(
        self,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        val_inputs: np.ndarray,
        val_targets: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15
    ) -> Dict:
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        print("\nTraining SNN with Surrogate Gradient BPTT...")
        print("=" * 60)
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_inputs, train_targets, batch_size)
            val_loss = self.validate(val_inputs, val_targets)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                best_weights = {
                    'W1': self.model.W1.copy(),
                    'b1': self.model.b1.copy(),
                    'W2': self.model.W2.copy(),
                    'b2': self.model.b2.copy(),
                    'W_out': self.model.W_out.copy(),
                    'b_out': self.model.b_out.copy()
                }
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        if best_weights:
            self.model.W1 = best_weights['W1']
            self.model.b1 = best_weights['b1']
            self.model.W2 = best_weights['W2']
            self.model.b2 = best_weights['b2']
            self.model.W_out = best_weights['W_out']
            self.model.b_out = best_weights['b_out']
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }


# ============================================================================
# Visualization
# ============================================================================

def visualize_surrogate_gradients():
    """Visualize and compare surrogate gradient functions."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    x = np.linspace(-3, 3, 1000)
    
    # Plot 1: Forward (Heaviside) vs Surrogates
    ax = axes[0]
    ax.plot(x, heaviside(x), 'k-', linewidth=2, label='Heaviside (forward)')
    ax.plot(x, 0.5 * (1 + np.tanh(2*x)), 'b--', linewidth=2, label='Tanh approx')
    ax.plot(x, 1/(1 + np.exp(-4*x)), 'r--', linewidth=2, label='Sigmoid approx')
    ax.set_xlabel('Membrane Potential (V - θ)')
    ax.set_ylabel('Output')
    ax.set_title('Spike Function Approximations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Surrogate Gradients
    ax = axes[1]
    for alpha in [1, 2, 5, 10]:
        ax.plot(x, atan_surrogate_gradient(x, alpha), 
                label=f'ATan (α={alpha})', linewidth=2)
    ax.set_xlabel('Membrane Potential (V - θ)')
    ax.set_ylabel('Surrogate Gradient')
    ax.set_title('ATan Surrogate Gradient (Paper\'s Choice)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Compare ATan vs Fast Sigmoid
    ax = axes[2]
    ax.plot(x, atan_surrogate_gradient(x, 2), 'b-', linewidth=2, 
            label='ATan (α=2)')
    ax.plot(x, fast_sigmoid_surrogate(x, 10), 'r--', linewidth=2,
            label='Fast Sigmoid')
    ax.set_xlabel('Membrane Potential (V - θ)')
    ax.set_ylabel('Surrogate Gradient')
    ax.set_title('Surrogate Gradient Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_snn_activity(model: SpikingNeuralNetwork, 
                           sample_input: np.ndarray,
                           sample_target: np.ndarray) -> plt.Figure:
    """Visualize SNN spike activity and predictions."""
    # Forward pass
    output, recordings = model.forward(sample_input.reshape(1, -1))
    
    spk1 = recordings['spk1'][:, 0, :]  # (n_steps, hidden_size)
    spk2 = recordings['spk2'][:, 0, :]
    mem2 = recordings['mem2'][:, 0, :]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Layer 1 spike raster
    ax = axes[0, 0]
    for neuron_id in range(min(50, spk1.shape[1])):
        spike_times = np.where(spk1[:, neuron_id] == 1)[0]
        ax.scatter(spike_times, np.full_like(spike_times, neuron_id), 
                   s=3, c='blue', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Neuron Index')
    ax.set_title('Hidden Layer 1 - Spike Raster (first 50 neurons)')
    
    # Layer 2 spike raster
    ax = axes[0, 1]
    for neuron_id in range(min(50, spk2.shape[1])):
        spike_times = np.where(spk2[:, neuron_id] == 1)[0]
        ax.scatter(spike_times, np.full_like(spike_times, neuron_id),
                   s=3, c='red', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Neuron Index')
    ax.set_title('Hidden Layer 2 - Spike Raster (first 50 neurons)')
    
    # Membrane potential traces
    ax = axes[1, 0]
    n_show = min(8, mem2.shape[1])
    for i in range(n_show):
        ax.plot(mem2[:, i], label=f'N{i}', alpha=0.7)
    ax.axhline(y=1.0, color='k', linestyle='--', label='Threshold', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Membrane Potential')
    ax.set_title('Layer 2 Membrane Potentials')
    ax.legend(fontsize=8, ncol=3)
    
    # Prediction vs Target
    ax = axes[1, 1]
    x_pos = np.arange(len(sample_target))
    width = 0.35
    ax.bar(x_pos - width/2, sample_target, width, label='Target', alpha=0.7, color='steelblue')
    ax.bar(x_pos + width/2, output[0], width, label='SNN Prediction', alpha=0.7, color='coral')
    ax.set_xlabel('Joint Index')
    ax.set_ylabel('Position (normalized)')
    ax.set_title('SNN Output vs Target')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'J{i+1}' for i in range(len(sample_target))])
    ax.legend()
    
    mse = np.mean((output[0] - sample_target) ** 2)
    ax.text(0.02, 0.98, f'MSE: {mse:.6f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Spiking Neural Network Activity Visualization', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# Main Demo
# ============================================================================

def demo_surrogate_gradient_training():
    """Complete demonstration of SNN training with surrogate gradients."""
    print("=" * 70)
    print("SURROGATE GRADIENT SANDBOX - SNN Training for Robot Dynamics")
    print("Pure NumPy Implementation (No PyTorch Required)")
    print("=" * 70)
    
    # Generate dataset
    print("\n[1/5] Generating robot arm dynamics dataset...")
    data_gen = RobotArmDataGenerator(n_joints=6)
    
    inputs, targets = data_gen.generate_dataset(
        n_trajectories=20,
        steps_per_trajectory=50
    )
    
    print(f"  Dataset size: {len(inputs)} samples")
    print(f"  Input shape: {inputs.shape} (torques + states)")
    print(f"  Target shape: {targets.shape} (next positions)")
    
    # Normalize data
    input_mean, input_std = inputs.mean(axis=0), inputs.std(axis=0) + 1e-8
    target_mean, target_std = targets.mean(axis=0), targets.std(axis=0) + 1e-8
    
    inputs_norm = (inputs - input_mean) / input_std
    targets_norm = (targets - target_mean) / target_std
    
    # Train/val split
    n_train = int(0.8 * len(inputs))
    train_inputs = inputs_norm[:n_train]
    train_targets = targets_norm[:n_train]
    val_inputs = inputs_norm[n_train:]
    val_targets = targets_norm[n_train:]
    
    print(f"  Train samples: {len(train_inputs)}")
    print(f"  Validation samples: {len(val_inputs)}")
    
    # Create SNN
    print("\n[2/5] Creating Spiking Neural Network...")
    lif_params = LIFParameters(
        beta=0.9,
        threshold=1.0,
        surrogate_alpha=2.0  # ATan steepness (paper's choice)
    )
    
    model = SpikingNeuralNetwork(
        input_size=18,    # 6 torques + 6 positions + 6 velocities
        hidden_size=64,   # Reduced for faster training
        output_size=6,
        n_steps=20,
        lif_params=lif_params
    )
    
    n_params = (model.W1.size + model.b1.size + 
                model.W2.size + model.b2.size +
                model.W_out.size + model.b_out.size)
    
    print(f"  Architecture: 18 -> 64 LIF -> 64 LIF -> 6")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Temporal steps: {model.n_steps}")
    print(f"  LIF beta (decay): {lif_params.beta}")
    print(f"  Surrogate: ATan (alpha={lif_params.surrogate_alpha})")
    
    # Visualize surrogate gradients
    print("\n[3/5] Visualizing surrogate gradient functions...")
    fig_surrogate = visualize_surrogate_gradients()
    fig_surrogate.savefig('surrogate_gradients.png', dpi=150, bbox_inches='tight')
    print("  Saved: surrogate_gradients.png")
    
    # Train
    print("\n[4/5] Training with Surrogate Gradient BPTT...")
    trainer = SNNTrainer(model, learning_rate=0.005, weight_decay=1e-4)
    
    results = trainer.train(
        train_inputs, train_targets,
        val_inputs, val_targets,
        n_epochs=25,
        batch_size=32,
        early_stopping_patience=8
    )
    
    print(f"\nFinal validation loss: {results['best_val_loss']:.6f}")
    
    # Plot training curves
    print("\n[5/5] Generating visualizations...")
    
    fig1, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results['train_losses'], label='Train Loss', linewidth=2)
    ax.plot(results['val_losses'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('SNN Training with ATan Surrogate Gradient BPTT', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig1.savefig('snn_training_curve.png', dpi=150, bbox_inches='tight')
    print("  Saved: snn_training_curve.png")
    
    # Visualize SNN activity
    sample_idx = np.random.randint(len(val_inputs))
    fig2 = visualize_snn_activity(model, val_inputs[sample_idx], val_targets[sample_idx])
    fig2.savefig('snn_activity_visualization.png', dpi=150, bbox_inches='tight')
    print("  Saved: snn_activity_visualization.png")
    
    # Compute test metrics
    pred, _ = model.forward(val_inputs[:100])
    mse = np.mean((pred - val_targets[:100]) ** 2)
    mae = np.mean(np.abs(pred - val_targets[:100]))
    
    # Denormalize for interpretable error
    pred_denorm = pred * target_std + target_mean
    target_denorm = val_targets[:100] * target_std + target_mean
    error_rad = np.abs(pred_denorm - target_denorm)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  MSE (normalized): {mse:.6f}")
    print(f"  MAE (normalized): {mae:.6f}")
    print(f"  Mean position error: {np.mean(error_rad):.6f} rad")
    print(f"  Max position error: {np.max(error_rad):.6f} rad")
    print(f"  Paper's β* bound: 0.021 rad")
    
    if np.mean(error_rad) < 0.021:
        print("\n  ✓ SNN achieves error below paper's theoretical bound!")
    else:
        print(f"\n  Note: More training or larger network may improve results")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM THIS EXPERIMENT")
    print("=" * 60)
    print("""
    1. SURROGATE GRADIENTS: The ATan function (paper's choice) provides
       smooth gradients where the true Heaviside step function has none.
       This enables backpropagation through spiking nonlinearities.
    
    2. TEMPORAL DYNAMICS: The LIF neurons integrate inputs over time,
       accumulating evidence before firing. This is crucial for processing
       time-varying robot dynamics.
    
    3. RATE CODING: The output is decoded by counting spikes over time,
       converting spike rates back to continuous position predictions.
    
    4. MEMBRANE LEAK (β): Controls how quickly neurons "forget" past inputs.
       β=0.9 means 90% of membrane potential persists each timestep.
    
    5. ENERGY EFFICIENCY: In hardware (Loihi, SpiNNaker), this network
       would only consume power when neurons spike - potentially 100-1000x
       more efficient than equivalent ANNs.
    """)
    
    plt.close('all')
    print("\nSurrogate Gradient SNN training complete!")
    
    return model, results


if __name__ == "__main__":
    np.random.seed(42)
    model, results = demo_surrogate_gradient_training()
