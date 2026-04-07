"""
Spike Encoder - Sensory Encoding Layer (SEL)
=============================================
Converts continuous joint position/error data into Poisson spike trains.

Based on the paper's description:
- Encoding firing rate: λᵢ(t) = λᵐₐˣ · |sᵢ(t)| / sᵐₐˣ
- Maximum firing rate: λᵐₐˣ = 500 Hz
- Higher joint error → higher frequency of spikes (1s) in bitstream

This implements the "Sensory Encoding Layer (SEL)" from the neuromorphic
control framework.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class PoissonSpikeEncoder:
    """
    Converts continuous signals (joint errors, velocities, forces) into
    Poisson-distributed spike trains suitable for SNN processing.
    """
    
    def __init__(
        self,
        lambda_max: float = 500.0,  # Maximum firing rate (Hz)
        signal_max: float = 1.0,     # Maximum expected signal magnitude
        dt: float = 0.001,           # Time step (1ms = 1kHz sampling)
        noise_threshold_factor: float = 1.5,  # Threshold = 1.5× noise floor
        noise_floor: float = 0.001   # Sensor noise floor
    ):
        """
        Initialize the Poisson Spike Encoder.
        
        Args:
            lambda_max: Maximum firing rate in Hz (paper uses 500 Hz)
            signal_max: Maximum expected signal magnitude for normalization
            dt: Simulation time step in seconds
            noise_threshold_factor: Multiplier for noise threshold
            noise_floor: Sensor noise floor (sub-noise → zero spikes)
        """
        self.lambda_max = lambda_max
        self.signal_max = signal_max
        self.dt = dt
        self.threshold = noise_threshold_factor * noise_floor
        
    def compute_firing_rate(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the instantaneous firing rate from signal magnitude.
        
        λᵢ(t) = λᵐₐˣ · |sᵢ(t)| / sᵐₐˣ
        
        Args:
            signal: Input signal array (can be scalar or vector)
            
        Returns:
            Firing rate array (clipped at lambda_max)
        """
        signal = np.asarray(signal)
        
        # Apply noise threshold - sub-noise signals produce zero spikes
        magnitude = np.abs(signal)
        magnitude = np.where(magnitude < self.threshold, 0, magnitude)
        
        # Compute firing rate proportional to signal magnitude
        rate = self.lambda_max * magnitude / self.signal_max
        
        # Clip to maximum firing rate to prevent spike flooding
        return np.clip(rate, 0, self.lambda_max)
    
    def generate_spikes(
        self,
        signal: np.ndarray,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Poisson spike train from a continuous signal.
        
        Args:
            signal: Input signal (scalar or 1D array for time series)
            duration: Duration of spike train in seconds
            
        Returns:
            spike_train: Binary array (1 = spike, 0 = no spike)
            time: Time array
        """
        n_steps = int(duration / self.dt)
        
        if np.isscalar(signal):
            # Static signal - constant firing rate
            firing_rate = self.compute_firing_rate(signal)
            firing_rates = np.full(n_steps, firing_rate)
        else:
            # Time-varying signal - interpolate to simulation resolution
            signal = np.asarray(signal)
            if len(signal) != n_steps:
                x_old = np.linspace(0, duration, len(signal))
                x_new = np.linspace(0, duration, n_steps)
                signal = np.interp(x_new, x_old, signal)
            firing_rates = self.compute_firing_rate(signal)
        
        # Poisson spike generation: P(spike) = λ · dt
        spike_probabilities = firing_rates * self.dt
        random_samples = np.random.random(n_steps)
        spike_train = (random_samples < spike_probabilities).astype(int)
        
        time = np.arange(n_steps) * self.dt
        
        return spike_train, time
    
    def encode_joint_errors(
        self,
        q_desired: np.ndarray,
        q_actual: np.ndarray,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode joint position errors into spike trains for all 6 joints.
        
        Args:
            q_desired: Desired joint positions (6,) or (T, 6)
            q_actual: Actual joint positions (6,) or (T, 6)
            duration: Duration in seconds
            
        Returns:
            spike_trains: Binary spike trains for each joint (n_steps, 6)
            time: Time array
        """
        errors = q_desired - q_actual
        
        n_steps = int(duration / self.dt)
        n_joints = 6
        
        spike_trains = np.zeros((n_steps, n_joints), dtype=int)
        
        for joint_idx in range(n_joints):
            if errors.ndim == 1:
                joint_error = errors[joint_idx]
            else:
                joint_error = errors[:, joint_idx]
                
            spike_trains[:, joint_idx], time = self.generate_spikes(
                joint_error, duration
            )
        
        return spike_trains, time


class PopulationEncoder:
    """
    Population coding encoder for force signals.
    Uses Gaussian receptive fields for finer resolution at low forces.
    
    From the paper: "each of the six force channels is encoded across
    eight LIF neurons with Gaussian receptive fields spanning the
    operational force range [0, 50] N"
    """
    
    def __init__(
        self,
        n_neurons: int = 8,
        value_range: Tuple[float, float] = (0, 50),
        lambda_max: float = 500.0,
        dt: float = 0.001
    ):
        """
        Initialize population encoder.
        
        Args:
            n_neurons: Number of neurons in the population
            value_range: (min, max) range of input values
            lambda_max: Maximum firing rate per neuron
            dt: Time step in seconds
        """
        self.n_neurons = n_neurons
        self.value_min, self.value_max = value_range
        self.lambda_max = lambda_max
        self.dt = dt
        
        # Create Gaussian receptive field centers
        self.centers = np.linspace(self.value_min, self.value_max, n_neurons)
        
        # Receptive field width (overlapping Gaussians)
        self.sigma = (self.value_max - self.value_min) / (n_neurons - 1) * 0.8
        
    def compute_population_rates(self, value: float) -> np.ndarray:
        """
        Compute firing rates for all neurons in the population.
        
        Args:
            value: Input value (e.g., force magnitude)
            
        Returns:
            Array of firing rates for each neuron
        """
        # Gaussian activation: stronger response near receptive field center
        activations = np.exp(-0.5 * ((value - self.centers) / self.sigma) ** 2)
        
        # Scale to firing rates
        rates = self.lambda_max * activations
        
        return rates
    
    def encode(
        self,
        values: np.ndarray,
        duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode a time series of values into population spike trains.
        
        Args:
            values: Input values (1D time series)
            duration: Duration in seconds
            
        Returns:
            spike_trains: (n_steps, n_neurons) binary spike array
            time: Time array
        """
        n_steps = int(duration / self.dt)
        values = np.asarray(values)
        
        if len(values) != n_steps:
            x_old = np.linspace(0, duration, len(values))
            x_new = np.linspace(0, duration, n_steps)
            values = np.interp(x_new, x_old, values)
        
        spike_trains = np.zeros((n_steps, self.n_neurons), dtype=int)
        
        for t in range(n_steps):
            rates = self.compute_population_rates(values[t])
            probs = rates * self.dt
            spike_trains[t] = (np.random.random(self.n_neurons) < probs).astype(int)
        
        time = np.arange(n_steps) * self.dt
        
        return spike_trains, time


def visualize_spike_raster(
    spike_trains: np.ndarray,
    time: np.ndarray,
    labels: Optional[list] = None,
    title: str = "Spike Raster Plot"
) -> plt.Figure:
    """
    Create a raster plot visualization of spike trains.
    
    Args:
        spike_trains: (n_steps, n_channels) binary spike array
        time: Time array
        labels: Channel labels
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    n_channels = spike_trains.shape[1]
    
    if labels is None:
        labels = [f"Joint {i+1}" for i in range(n_channels)]
    
    fig, axes = plt.subplots(n_channels + 1, 1, figsize=(12, 2 * n_channels + 2), 
                             sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    
    for i in range(n_channels):
        spike_times = time[spike_trains[:, i] == 1]
        axes[i].eventplot([spike_times], colors=[colors[i]], lineoffsets=0)
        axes[i].set_ylabel(labels[i], fontsize=10)
        axes[i].set_yticks([])
        axes[i].set_xlim([time[0], time[-1]])
        
        # Show firing rate
        window_ms = 10  # 10ms window for rate estimation
        window_samples = int(window_ms / 1000 / (time[1] - time[0]))
        if window_samples > 0:
            kernel = np.ones(window_samples) / (window_ms / 1000)
            rate = np.convolve(spike_trains[:, i], kernel, mode='same')
            ax2 = axes[i].twinx()
            ax2.plot(time, rate, color=colors[i], alpha=0.3, linewidth=0.8)
            ax2.set_ylabel('Hz', fontsize=8)
    
    # Combined raster in last subplot
    for i in range(n_channels):
        spike_times = time[spike_trains[:, i] == 1]
        axes[-1].eventplot([spike_times], colors=[colors[i]], 
                          lineoffsets=i, linelengths=0.8)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    axes[-1].set_ylabel('Channel', fontsize=10)
    axes[-1].set_yticks(range(n_channels))
    axes[-1].set_yticklabels(labels)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def demo_spike_encoding():
    """
    Demonstrate spike encoding with simulated joint errors.
    Shows how higher errors produce higher spike rates.
    """
    print("=" * 60)
    print("SPIKE ENCODER DEMO - Sensory Encoding Layer (SEL)")
    print("=" * 60)
    
    # Initialize encoder with paper parameters
    encoder = PoissonSpikeEncoder(
        lambda_max=500.0,  # 500 Hz max rate from paper
        signal_max=0.5,    # Max error in radians (~28 degrees)
        dt=0.001           # 1kHz sampling
    )
    
    duration = 0.5  # 500ms demonstration
    t = np.linspace(0, duration, 500)
    
    # Simulate 6-DOF joint errors with varying magnitudes
    # Higher error = higher spike rate
    joint_errors = np.zeros((len(t), 6))
    
    # Joint 1: Constant small error (0.02 rad) → low spike rate
    joint_errors[:, 0] = 0.02
    
    # Joint 2: Constant medium error (0.1 rad) → medium spike rate  
    joint_errors[:, 1] = 0.1
    
    # Joint 3: Constant large error (0.3 rad) → high spike rate
    joint_errors[:, 2] = 0.3
    
    # Joint 4: Sinusoidal error → varying spike rate
    joint_errors[:, 3] = 0.15 * np.sin(2 * np.pi * 5 * t)
    
    # Joint 5: Step change in error
    joint_errors[:, 4] = np.where(t < 0.25, 0.05, 0.25)
    
    # Joint 6: Decaying error (like settling after disturbance)
    joint_errors[:, 5] = 0.3 * np.exp(-10 * t)
    
    print(f"\nSimulating {duration*1000:.0f}ms of joint error encoding...")
    print(f"Time step: {encoder.dt*1000:.1f}ms ({int(1/encoder.dt)} Hz sampling)")
    print(f"Max firing rate: {encoder.lambda_max} Hz")
    
    # Encode each joint
    n_steps = int(duration / encoder.dt)
    spike_trains = np.zeros((n_steps, 6), dtype=int)
    
    for joint_idx in range(6):
        spike_trains[:, joint_idx], time = encoder.generate_spikes(
            joint_errors[:, joint_idx], duration
        )
    
    # Analyze results
    print("\n" + "-" * 50)
    print("ENCODING RESULTS:")
    print("-" * 50)
    
    labels = [
        "J1: Small const (0.02 rad)",
        "J2: Medium const (0.1 rad)", 
        "J3: Large const (0.3 rad)",
        "J4: Sinusoidal",
        "J5: Step change",
        "J6: Exponential decay"
    ]
    
    for i in range(6):
        n_spikes = np.sum(spike_trains[:, i])
        avg_rate = n_spikes / duration
        theoretical_rate = encoder.compute_firing_rate(
            np.mean(np.abs(joint_errors[:, i]))
        )
        print(f"{labels[i]}")
        print(f"  Total spikes: {n_spikes:4d} | Avg rate: {avg_rate:6.1f} Hz | "
              f"Expected: ~{theoretical_rate:.1f} Hz")
    
    # Create visualization
    print("\nGenerating spike raster visualization...")
    
    fig = visualize_spike_raster(spike_trains, time, labels, 
                                  "Poisson Spike Encoding: Joint Errors → Spike Trains")
    
    # Save figure
    fig.savefig('spike_encoder_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: spike_encoder_demo.png")
    
    # Also create firing rate vs error plot
    fig2, ax = plt.subplots(figsize=(10, 5))
    
    errors = np.linspace(0, 0.5, 100)
    rates = encoder.compute_firing_rate(errors)
    
    ax.plot(errors * 180 / np.pi, rates, 'b-', linewidth=2)
    ax.axhline(y=encoder.lambda_max, color='r', linestyle='--', 
               label=f'λ_max = {encoder.lambda_max} Hz')
    ax.axvline(x=encoder.threshold * 180 / np.pi, color='g', linestyle='--',
               label=f'Noise threshold')
    
    ax.set_xlabel('Joint Error (degrees)', fontsize=12)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax.set_title('Sensory Encoding: Error Magnitude → Firing Rate', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig2.savefig('firing_rate_curve.png', dpi=150, bbox_inches='tight')
    print("Saved: firing_rate_curve.png")
    
    # Population coding demo for force signals
    print("\n" + "=" * 60)
    print("POPULATION CODING DEMO - Force Signals")
    print("=" * 60)
    
    pop_encoder = PopulationEncoder(
        n_neurons=8,
        value_range=(0, 50),
        lambda_max=500.0
    )
    
    # Show receptive fields
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    forces = np.linspace(0, 50, 200)
    for i in range(pop_encoder.n_neurons):
        rates = [pop_encoder.compute_population_rates(f)[i] for f in forces]
        ax1.plot(forces, rates, label=f'Neuron {i+1}', linewidth=2)
    
    ax1.set_xlabel('Force (N)', fontsize=12)
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax1.set_title('Population Coding Receptive Fields', fontsize=14)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Encode a force signal
    force_signal = 25 + 15 * np.sin(2 * np.pi * 3 * t)  # 3Hz oscillation around 25N
    pop_spikes, pop_time = pop_encoder.encode(force_signal, duration)
    
    for i in range(pop_encoder.n_neurons):
        spike_times = pop_time[pop_spikes[:, i] == 1]
        ax2.eventplot([spike_times], lineoffsets=i, linelengths=0.8,
                      colors=[plt.cm.viridis(i / pop_encoder.n_neurons)])
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, force_signal, 'r-', alpha=0.5, label='Force signal')
    ax2_twin.set_ylabel('Force (N)', color='r')
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Neuron Index', fontsize=12)
    ax2.set_yticks(range(pop_encoder.n_neurons))
    ax2.set_title('Population Spike Encoding of Force Signal', fontsize=14)
    
    fig3.tight_layout()
    fig3.savefig('population_coding_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: population_coding_demo.png")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT FROM THE PAPER:")
    print("=" * 60)
    print("""
The Sensory Encoding Layer (SEL) converts continuous joint tracking errors
eᵢ = qᵢᵈ - qᵢ into Poisson-distributed spike trains where:

    λᵢ(t) = λᵐₐˣ · |sᵢ(t)| / sᵐₐˣ

This means:
  • HIGHER error → HIGHER firing rate → MORE spikes
  • ZERO error → ZERO spikes → ZERO power consumption!
  
This is the key to neuromorphic efficiency: the chip only computes
when there's something to correct. At steady state, power draw is
essentially zero - achieving the paper's 2,244× power reduction.
""")
    
    plt.show()


if __name__ == "__main__":
    demo_spike_encoding()
