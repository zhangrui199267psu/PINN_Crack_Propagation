# X-PINN Crack Propagation (Full Codebase + Concise Flow)

You asked for **no feature loss** and a **concise workflow**. This repository now keeps the full original modules while giving a short, practical run flow.

## 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Fast / Medium / Production Runs

```bash
# Fast sanity run
python run_experiments.py --config configs/mode1_quick.yaml

# Medium run
python run_experiments.py --config configs/mode1_medium.yaml

# Production run
python run_experiments.py --config configs/mode1_production.yaml
```

## 3) Outputs

After each run you get:
- `stress_field.png`
- `loss_history.png`
- `ki_path_independence.png`
- `ki_extrapolation.png`

Console output also prints:
- analytical and PINN-estimated `K_I`
- path independence metrics
- Williams-series validation summary.

## 4) Concise code flow (full features)

1. `run_experiments.py`
   - Loads YAML config
   - Builds geometry/material/training objects
   - Trains solver
   - Computes stress + K_I + validation
   - Saves plots
2. `xpinn_fracture/solver.py`
   - Builds PDE dataset and BCs
   - Trains Adam (+ optional L-BFGS)
   - Optional RAD adaptive resampling
3. `xpinn_fracture/networks.py`
   - 3-branch X-PINN decomposition (`u_C + H*u_D + psi*u_S`)
   - XFEM crack-tip enrichment features
4. `xpinn_fracture/physics.py`
   - Linear elasticity residual (`div(sigma)=0`)
   - Domain BCs + crack-face traction-free BCs
5. `xpinn_fracture/postprocess.py` + `validation.py`
   - stress components/von Mises
   - J-integral and displacement extrapolation `K_I`
   - analytical/Williams comparison
6. `xpinn_fracture/visualization.py`
   - all result figures.

## 5) Files kept intentionally

- Full original framework (`xpinn_fracture/`)
- Experiment runner (`run_experiments.py`)
- Legacy entry (`xfem_crack.py`)
- Three configs (`quick`, `medium`, `production`)
- `minimal_pinn/` personal mini-example (optional)

If you only want the concise run path, use Section 2 above.
