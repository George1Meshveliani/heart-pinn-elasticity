# 🫀 Heart PINN: Estimating Cardiac Elasticity with Physics-Informed Neural Networks

This repository presents a **Physics-Informed Neural Network (PINN)** framework for estimating a **stiffness-like parameter (β)** in a cardiovascular-inspired dynamical system.

The method integrates:

* physical laws (ODE)
* noisy and sparse observations
* bounded physiological constraints

---

## 📄 Research Paper

**Constrained Physics-Informed Neural Networks for Estimating a Stiffness-Like Cardiovascular Parameter**
*George Meshveliani*

This project accompanies the research paper submitted to arXiv.

---

## 🔬 Problem Overview

We model a simplified cardiovascular system as a **forced damped oscillator**:

ẍ + αẋ + βx = A sin(2π f t)

Where:

* β → stiffness / elasticity (**unknown, to be learned**)
* α → damping coefficient (known)
* A, f → forcing parameters (known)

---

## 💡 Key Idea

Instead of learning β freely, we impose **physiological constraints**:

β ∈ [β_min, β_max]

This transforms the problem into a **constrained inverse learning problem**, improving:

* stability
* interpretability
* physical realism

---

## 🧠 Methodology

The PINN is trained using a combined loss function:

* data loss (x)
* velocity loss (ẋ)
* physics loss (ODE residual)
* initial condition loss

Final loss:

L = 10 L_x + 10 L_v + 10 L_physics + 20 L_IC

---

## 📊 Results

| Quantity  | Value     |
| --------- | --------- |
| True β    | 16.0      |
| Learned β | 15.209    |
| Error (%) | **4.94%** |
| RMSE (x)  | 0.036     |
| RMSE (ẋ)  | 0.197     |

✅ Accurate parameter recovery under:

* noisy data
* sparse observations

---

## ▶️ Run Experiment

python heart_pinn_bounded_beta.py

---

## 📈 Output

The script will:

* train the PINN model
* estimate parameter β
* print performance metrics
* generate plots:

  * signal reconstruction
  * beta convergence
  * training loss curves

---

## 🧪 Reproducibility

* fixed random seeds
* deterministic setup
* synthetic ground-truth data

---

## 📌 Key Contributions

* Constrained PINN for parameter estimation
* Physiologically bounded inverse learning
* Integration of derivative observations (ẋ)
* Stable recovery under noisy conditions

---

## ⚠️ Limitations

* simplified oscillator model (not full cardiovascular dynamics)
* synthetic data only
* single parameter estimation

---

## 🔮 Future Work

* application to real physiological datasets
* multi-parameter estimation
* more realistic cardiovascular models
* clinical validation

---

## 📚 References

* Raissi, Perdikaris, Karniadakis (2019) — Physics-Informed Neural Networks
* Karniadakis et al. (2021) — Physics-Informed Machine Learning
* Cuomo et al. (2022) — Scientific Machine Learning via PINNs
* Wang et al. (2020) — Gradient Pathologies in PINNs
* Wang et al. (2023) — Expert Guide to Training PINNs
* Rackauckas et al. (2020) — Universal Differential Equations
* Brunton et al. (2016) — SINDy (Equation Discovery)
* Rudy et al. (2017) — PDE Discovery from Data
* Quarteroni et al. (2000) — Cardiovascular Fluid Dynamics
* Nichols & O’Rourke (2011) — Blood Flow in Arteries

---

## 📬 Contact

George Meshveliani
📍 Tbilisi, Georgia
📧 gmeshveliani@cu.edu.ge

---


