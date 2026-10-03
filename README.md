# 1D Heat Equation: NumPy solver, MATLAB driver, validated on NASA Li-ion data

## Files
| File | What it does |
|---|---|
| `heat1d.py` | NumPy solver for the 1D heat equation. Supports Cartesian or cylindrical (radial) geometry; explicit, implicit and Crank–Nicolson time stepping; Dirichlet, Neumann or Robin (convection) boundaries; a heat source; and batching many independent runs. `python heat1d.py` runs self-checks against exact solutions. |
| `heat_equation_1d.m` | Demo (heated aluminum rod, animated) and convergence check (`testCase = 'verify'`). |
| `validate_battery_thermal.m` | Fits and validates a radial thermal model of an 18650 cell against the NASA battery data. |
| `data/` | `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` (NASA PCoE Battery Aging, set 1). |
| `results/<heatModel>/` | Figures, `validation_summary.csv` and `fitted_parameters.mat` from the last run. |

MATLAB calls the solver through its Python interface. Check with `pyenv` that MATLAB is using a Python that has NumPy installed.

## Battery thermal model
Radial heat conduction in a cylindrical cell (R = 9 mm), with uniform internal heating:

```
dT/dt = alpha (1/r) d/dr (r dT/dr) + beta q(t) / (rho c V)
dT/dr = 0 at r = 0,     -k dT/dr = h_eff (T - T_amb) at r = R
```

- **Heat source** (`heatModel = "voltage"`, the default): irreversible heat `q = |I| (U_ocv - V)`.
  - The open-circuit voltage `U_ocv(SOC)` is estimated for each cycle from the nearest *full* charge. At equal state of charge, `U = (I_dis V_ch + I_ch V_dis) / (I_ch + I_dis)`.
  - Partial top-up charges are skipped because they corrupt this estimate.
- **Alternative** (`heatModel = "eis"`): `q = I^2 (Re + Rct)`, using the most recent impedance test.
- **Fixed properties** (typical literature values): mass 45 g, c = 1000 J/(kg K), radial k = 0.20 W/(m K). End caps are folded into `h_eff = h (1 + R/H)`.
- **Fitted parameters:** `h` and `beta`, fitted with `fminsearch` on every discharge of B0005–B0007 (504 cycles).
- **Validation:** B0018 (132 cycles) is predicted **blind**, with no refitting. The model's surface temperature is compared with the thermocouple reading.

## Results (last run)
| Heat model | Fitted h [W/m²K] | beta | B0018 RMSE | B0018 mean peak error | Train RMSE (B0005/6/7) |
|---|---|---|---|---|---|
| voltage (default) | 5.0 | 0.55 | **1.00 °C** | **0.58 °C** | 0.96 / 1.28 / 1.05 °C |
| eis | 4.4 | 0.68 | 2.18 °C | 1.12 °C | 1.14 / 1.55 / 1.29 °C |

The voltage-based heat source captures the heating spike at the end of discharge. It also captures the rise in peak temperature as the cells age, and does so without being told about aging. The EIS-based source misses both.

## Known limitations
- **B0005 and B0007 run hotter than predicted:** their measured peaks are 2–3 °C above the model. B0005–B0007 were cycled in lockstep (identical timing) and probably shared a fixture, so neighboring cells may have heated each other.
- **Cooling after cutoff:** after the load turns off, the measured surface cools faster than the model. A higher effective h, or less thermal mass near the surface, would explain it.
- **beta ≈ 0.55:** either the averaged-voltage `U_ocv` overestimates heat (charge/discharge hysteresis), or the effective heat capacity is higher than assumed. Only the ratio beta/(rho c) is well constrained.
- **Entropic heat** is not modeled. It likely explains the small temperature bump about 10 minutes into each discharge.

## Running
```matlab
validate_battery_thermal                      % voltage-based heat (default)
heatModel = "eis"; validate_battery_thermal   % EIS-based heat
heat_equation_1d                              % rod demo
testCase = 'verify'; heat_equation_1d         % convergence check
```
A full validation run takes about 1 minute, since all 504 training cycles are solved in one batched call.

## Data citation
B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
