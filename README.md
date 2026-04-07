<div align="center">

#  Neuromorphic SNN Controller

### A biologically-inspired control framework for robotic arm systems using Spiking Neural Networks - trained with surrogate gradient backpropagation and verified via Lyapunov stability analysis, paired with a real-time Next.js visualization portal.

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind_CSS-4.x-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-7F77DD?style=for-the-badge)

</div>

<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&pause=1000&color=7F77DD&center=true&vCenter=true&width=600&lines=Spiking+Neural+Networks+%7C+Neuromorphic+Computing;Surrogate+Gradient+Training+%7C+BPTT;Lyapunov+Stability+Analysis;Next.js+Visualization+Portal" alt="Typing animation" />
</div>

##  What is this?

Traditional robot controllers rely on continuous, rate-coded signals. This project takes a different approach - it mimics how biological neural systems actually work, using **sparse spike events** instead of continuous activations.

The result is a controller that is:
-  **Energy-efficient** - spikes fire only when needed, not continuously
-  **Mathematically stable** - Lyapunov theory guarantees convergence
-  **Research-grounded** - implements the Sensory Encoding Layer (SEL) and Hierarchical Processing Layer (HPL) architecture from neuromorphic control literature
-  **Visualizable** - a full Next.js portal to observe neural firing patterns in real time

---

##  Architecture

```
  ┌─────────────────────────────────────────────────────┐
  │                   INPUT LAYER                        │
  │         Joint position error  ·  Velocity           │
  └────────────────────┬────────────────────────────────┘
                       │
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │            SENSORY ENCODING LAYER  (SEL)             │
  │                                                      │
  │   λᵢ(t) = λᵐᵃˣ · |sᵢ(t)| / sᵐᵃˣ                  │
  │   λᵐᵃˣ = 500 Hz  ·  Poisson spike trains            │
  └────────────────────┬────────────────────────────────┘
                       │  spike trains  {0,1}ᵀ
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │         HIERARCHICAL PROCESSING LAYER  (HPL)         │
  │                                                      │
  │   Leaky Integrate-and-Fire neurons                   │
  │   Trained via BPTT · ATan surrogate gradient         │
  └────────────────────┬────────────────────────────────┘
                       │  predicted torques
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │              LYAPUNOV STABILITY CHECK                │
  │                                                      │
  │   V(x) > 0  ·  dV/dt < 0  →  asymptotic stability  │
  └────────────────────┬────────────────────────────────┘
                       │
                       ▼
                 CONTROL OUTPUT  
```

---

##  Repository Structure

```
neuromorphic-snn-controller/
│
├──  scripts/
│   ├── spike_encoder.py            # SEL — Poisson spike train generation
│   ├── snn_surrogate_gradient.py   # SNN training with ATan surrogate BPTT
│   ├── lyapunov_stability.py       # Energy-based stability verification
│   ├── run_all_experiments.py      # Unified experiment runner
│   └── pyproject.toml              # Python dependency manifest
│
├──  app/
│   ├── page.tsx                    # Neural firing visualization portal
│   └── layout.tsx
│
├──  components/                  # shadcn/ui + Radix UI component library
├──  lib/                         # Shared TypeScript utilities
├──  public/                      # Static assets
├──  hooks/                       # Custom React hooks
│
├── package.json
├── tsconfig.json
├── next.config.mjs
└── README.md
```

---

##  System Overview

<div align="center">

<img src="./Neuromorphic Control Systems.png" alt="Neuromorphic Control Systems - System Overview" width="85%" />

<sub><i>Figure: Full neuromorphic control pipeline - from sensory encoding to stable motor output</i></sub>

</div>

---

##  Tech Stack

<div align="center">

| Domain | Technology | Purpose |
|:---|:---|:---|
| **SNN Core** | Python 3.10+ · NumPy | Spike encoding, LIF neurons, BPTT |
| **Training** | Custom ATan surrogate gradient | Differentiable spike approximation |
| **Stability** | Lyapunov analysis | Mathematical convergence guarantee |
| **Frontend** | Next.js 16 · React 19 | Real-time visualization portal |
| **Styling** | Tailwind CSS 4 · shadcn/ui | Component design system |
| **Charts** | Recharts · Matplotlib | Spike raster plots & learning curves |
| **Package mgmt** | pnpm · uv | Fast, reproducible installs |

</div>

---

##  Installation

### Python - SNN Scripts

```bash
# Recommended: using uv
uv sync

# Alternative: pip
pip install numpy matplotlib
```

### Node - Visualization Portal

```bash
pnpm install
pnpm dev
# → http://localhost:3000
```

---

##  Usage

```bash
# 1. Run the spike encoder
uv run python scripts/spike_encoder.py

# 2. Train the SNN with surrogate gradients
uv run python scripts/snn_surrogate_gradient.py

# 3. Verify Lyapunov stability
uv run python scripts/lyapunov_stability.py

# 4. Run all experiments end-to-end
uv run python scripts/run_all_experiments.py
```

> Each script outputs `.png` visualization files to the working directory.

---

##  Key Parameters

| Parameter | Value | Description |
|:---|:---:|:---|
| `lambda_max` | `500 Hz` | Maximum Poisson firing rate |
| Surrogate function | `ATan` | Smooth approximation of the Heaviside step |
| Training method | `BPTT` | Backpropagation through time |
| Stability criterion | `Lyapunov` | V(x) > 0, dV/dt < 0 |
| Rate encoding | `λᵢ ∝ \|error\|` | Higher error → denser spikes |

---

##  Background Reading

If you're new to spiking neural networks or neuromorphic computing, these concepts are central to this project:

- **LIF Neurons** - Leaky Integrate-and-Fire: the standard neuron model for SNNs
- **Surrogate Gradients** - enable backprop through non-differentiable spike functions
- **Lyapunov Stability** - energy function approach to proving a system won't diverge
- **Poisson Spike Trains** - stochastic encoding of continuous signals as binary spike events

---

##  Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-idea`)
5. Open a Pull Request

---

##  License

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details.

---

##  Contact

For research collaboration or questions, open an issue or reach out via GitHub.

---

<div align="center">
<sub>shniharard@gmail.com · April 2026 · Under Development</sub>
</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=7F77DD&height=80&section=footer" alt="footer" width="100%"/> <sub>Last updated: April 2026 · Nihara Dayarathne</sub>
