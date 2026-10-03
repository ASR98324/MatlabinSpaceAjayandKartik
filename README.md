# Mars Habitat Autonomous Power Allocation (MATLAB in Space, Track 2)

**Hook:** a Mars habitat that uses its own walls as a battery, storing heat when solar is plentiful so it can cut heater power through a dust storm while keeping life support running.

## Problem
TODO: 2-3 sentences. Habitat 20 light-minutes from Earth; dust storm + aging battery + string failure; no human in the loop.

## Approach
- **Heat equation (PDE):** 1D conduction through the wall + lumped cabin node, backward Euler. Linear in heater power, so it fits inside the optimizer. (`habitat/thermal.py`)
- **Fourier forecast + storm detection:** harmonic regression on past sols only; storm flagged when actual energy drops below forecast by a threshold learned from past forecast errors. (`habitat/forecast.py`)
- **Controllers:** naive, rule-based priority shedding (baseline), thermal-aware MPC (TODO). (`habitat/controllers.py`)

## Data: real vs simulated
| Input | Source | Real? |
|---|---|---|
| Battery capacity fade | NASA PCoE Li-ion Battery Aging (B0005), 1 cycle = 1 sol | Real |
| Dust storm opacity | Shaped on 2018 global dust storm (Opportunity) | Real values, stylized profile |
| Solar irradiance at Mars | Mean solar constant at Mars | Real constant |
| Thermal parameters, loads | Assumptions in `habitat/config.py` | Simulated |

## How to run
```bash
pip install -r requirements.txt
# put the NASA battery .mat files anywhere under ./data
python run_scenario.py
python run_scenario.py --tau-peak 10.8   # full 2018-strength storm
```

## Results
TODO: timeline plot, comparison table, Monte Carlo results, sensitivity analysis.

## Limitations and next steps
- NASA cells were cycled at room temperature in a lab, not Mars cold.
- Thermal and load parameters are assumptions (see sensitivity analysis).
- Excess heat above 26 C is assumed dumped by passive radiator louvers.
- Next: feed a real telemetry anomaly detector (e.g. SMAP) into the failure flags.

## Dataset credits
- NASA Ames Prognostics Center of Excellence, Li-ion Battery Aging Data Set.
- TODO: cite 2018 dust storm opacity source.
