# Minimal PINN Cracking (2 files)

This branch keeps only minimal personal PINN cracking code:

- `minimal_pinn/model.py` — tiny MLP model `CrackPINN`
- `minimal_pinn/train.py` — PDE loss + BC loss + crack jump loss + Adam loop

## Run

```bash
python minimal_pinn/train.py
```

## Notes

- This is intentionally minimal for quick experimentation.
- Domain is unit square with simple Mode-I-like boundary constraints.
