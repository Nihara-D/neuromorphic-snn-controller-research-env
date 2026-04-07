#!/usr/bin/env python3
"""
Neuromorphic Robotics Experiments Runner
=========================================
Run all three experiments from the paper:

1. Spike Encoder (Sensory Encoding Layer)
2. Surrogate Gradient SNN Training
3. Lyapunov Stability Visualization

Usage:
    uv run run_all_experiments.py [--experiment N]

Where N is:
    1 = Spike Encoder only
    2 = SNN Training only
    3 = Lyapunov Analysis only
    (no flag = run all)
"""

import sys
import argparse


def run_spike_encoder():
    """Run the Poisson spike encoding demonstration."""
    print("\n" + "█" * 70)
    print("█ EXPERIMENT 1: SPIKE ENCODER                                        █")
    print("█" * 70)
    
    from spike_encoder import demo_spike_encoding
    demo_spike_encoding()


def run_snn_training():
    """Run the surrogate gradient SNN training demonstration."""
    print("\n" + "█" * 70)
    print("█ EXPERIMENT 2: SURROGATE GRADIENT SNN TRAINING                      █")
    print("█" * 70)
    
    from snn_surrogate_gradient import demo_surrogate_gradient_training
    demo_surrogate_gradient_training()


def run_lyapunov_analysis():
    """Run the Lyapunov stability analysis demonstration."""
    print("\n" + "█" * 70)
    print("█ EXPERIMENT 3: LYAPUNOV STABILITY ANALYSIS                          █")
    print("█" * 70)
    
    from lyapunov_stability import demo_lyapunov_stability
    demo_lyapunov_stability()


def main():
    parser = argparse.ArgumentParser(
        description="Run neuromorphic robotics experiments"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=int,
        choices=[1, 2, 3],
        help="Run specific experiment (1=Spike, 2=SNN, 3=Lyapunov)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Don't show interactive plots (still saves figures)"
    )
    
    args = parser.parse_args()
    
    if args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
    
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║   NEUROMORPHIC CONTROL FOR 6-DOF ROBOTIC MANIPULATORS              ║")
    print("║   Implementation of Key Paper Concepts                              ║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\nPaper: 'Neuromorphic Control of 6-DOF Industrial Robotic Arms'")
    print("       'Spiking Neural Networks with Lyapunov Stability Guarantees'")
    print("\nThree key experiments:")
    print("  1. Spike Encoder - Convert joint errors → Poisson spike trains")
    print("  2. SNN Training - Surrogate gradient BPTT with snntorch")
    print("  3. Lyapunov Analysis - Verify stability under disturbances")
    
    if args.experiment == 1:
        run_spike_encoder()
    elif args.experiment == 2:
        run_snn_training()
    elif args.experiment == 3:
        run_lyapunov_analysis()
    else:
        # Run all experiments
        run_spike_encoder()
        run_snn_training()
        run_lyapunov_analysis()
    
    print("\n" + "═" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("═" * 70)
    print("\nGenerated output files:")
    print("  • spike_encoder_demo.png - Spike raster plots")
    print("  • firing_rate_curve.png - Rate vs error relationship")
    print("  • population_coding_demo.png - Force signal encoding")
    print("  • snn_training_curve.png - Training loss curves")
    print("  • snn_activity_visualization.png - Network activity")
    print("  • lyapunov_stability_analysis.png - V(t) and error bounds")
    print("  • phase_portraits.png - Error convergence")
    print("  • lyapunov_step_disturbance.png - Sustained disturbance test")


if __name__ == "__main__":
    main()
