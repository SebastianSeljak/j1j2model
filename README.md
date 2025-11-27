# Neural Quantum States: Solving the J1-J2 Heisenberg Model with CNNs

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

A machine learning project exploring the intersection of Deep Learning and Quantum Many-Body Physics. We utilize 1D Convolutional Neural Networks within a Variational Monte Carlo (VMC) framework to approximate the ground states of frustrated quantum spin chains.

**Course:** Math 156 - Machine Learning (Fall 2025)

---

## 🌌 Project Overview

Simulating quantum systems is computationally expensive because the state space grows exponentially with system size ($2^N$). This "Curse of Dimensionality" prevents exact simulation of systems with more than $\approx 40$ particles.

This project uses **Neural Quantum States (NQS)** to compress the wavefunction into a neural network. We focus on the **1D $J_1-J_2$ Heisenberg Model**, a system known for "geometric frustration" and complex phase transitions.

We implement and compare three approaches:
1.  **Exact Diagonalization:** A brute-force physics engine to generate ground truth labels (limited to small $N$).
2.  **Supervised Learning:** A CNN trained to regress energy from spin configurations.
3.  **Unsupervised VMC:** A CNN trained via Reinforcement-style learning to minimize the energy expectation $\langle H \rangle$ directly, without labels.

---

## 📂 Repository Structure

```text
.
├── models/
│   ├── cnn.py                  # Standard Real-Valued 1D CNN
│   └── complex_cnn.py          # Complex-Valued CNN (Log-Amp + Phase heads)
├── physics/
│   ├── j1j2_solver.py          # Exact Diagonalization Solver (Hamiltonian matrix construction)
│   └── j1j2_unsupervised.py    # Local Energy and Marshall Sign Rule logic
├── train/
│   ├── train_cnn.py            # Supervised training loop
│   └── train_vmc.py            # Unsupervised VMC optimization loop (Gradient estimation)
├── utils/
│   ├── mcmc_sampler.py         # Metropolis-Hastings sampling algorithm
│   └── plotting.py             # Visualization tools for loss and spin textures
├── run_unsupervised.ipynb      # MAIN NOTEBOOK: Runs the full VMC experiment
├── hyperparam_tune.ipynb       # Grid search for architecture optimization
├── large_n_system.ipynb        # Scalability experiments (N=16)
├── report.pdf                  # Final academic report
└── README.md
```

---

## 🚀 Getting Started

### 1. Generating Ground Truth (Small Systems)
To verify the physics engine and see the exact ground state energy for a system of size $N=10$ at the frustrated point $J_2=0.5$:

```python
from physics.j1j2_solver import QuantumJ1J2Solver
solver = QuantumJ1J2Solver(n_spins=10, J1=1.0, J2=0.5)
print(f"Exact Ground State Energy: {solver.ground_state_energy}")
```

### 2. Running the VMC Solver
The core of the project is in `run_unsupervised.ipynb`.
1.  **Initialization:** Sets up the Complex CNN.
2.  **Sampling:** Uses Markov Chain Monte Carlo (MCMC) to sample spin states from $|\psi|^2$.
3.  **Optimization:** Minimizes energy using the log-derivative gradient estimator.

### 3. Hyperparameter Tuning
Run `hyperparam_tune.ipynb` to perform a grid search over Learning Rate, Batch Size, and Kernel Size for an $N=8$ system.

---

## 🧠 Model Architectures

We explore two distinct architectures to handle the quantum wavefunction $\Psi(S)$.

### Real-Valued CNN
*   **Output:** Single scalar (Real).
*   **Logic:** Represents amplitude only. Requires manual "Marshall Sign Rule" injection to handle quantum signs.
*   **Failure Mode:** Fails when $J_2 > 0$ (frustrated systems) because the manual sign rule becomes invalid.

### Complex-Valued CNN
*   **Output:** Two scalars (Real part $\ln|A|$ and Imaginary part $\phi$).
*   **Logic:** $\Psi(S) = e^{\ln|A| + i\phi}$.
*   **Success Mode:** Learns to rotate phase angles automatically to minimize energy, solving the "Sign Problem" without prior physics knowledge.

---

## 📊 Key Results

### 1. The Sign Problem
We compared both models on an $N=10$ chain.
*   **Real CNN:** Converged rapidly with help of sign rule.
*   **Complex CNN:** Converged to ~$-3.0$~ after initial plateau
*   **Conclusion:** Complex weights are necessary for frustrated quantum systems.

### 2. Kernel Size
Hyperparameter tuning revealed that a **Kernel Size of 5** outperforms smaller kernels for frustrated systems, suggesting the network learns non-local correlations beyond immediate neighbors.

### 3. Large System Scaling
We successfully trained a Complex CNN on an $N=16$ system.
*   **Texture Plot:** The heatmap below (generated in `large_n_system.ipynb`) shows the spin configuration stabilizing into a dimerized pattern over time, consistent with the Majumdar-Ghosh phase.

---

## 👥 Contributors

*   **Sebastian:** Complex-Valued CNN architecture, Hyperparameter Tuning, Large $N$ visualization.
*   **Julieanna:** Unsupervised Real CNN implementation, Metropolis-Hastings (MCMC) sampling.
*   **Yunhan:** Unsupervised training logic and debugging.
*   **Ann:** Supervised Learning baseline and data validation.
*   **Zhihao:** Exact Hamiltonian Solver (Physics Engine).
*   **TM:** Physics theory, mathematical derivations, and VMC gradient logic.

## 📚 References
1.  Carleo, G., & Troyer, M. (2017). *Solving the quantum many-body problem with artificial neural networks*. Science.
2.  Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.