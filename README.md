# Walls as a Battery: Autonomous Power Allocation for a Mars Habitat

**MATLAB in Space Hackathon, Track 2: Energy, Power Grid & Closed-Loop Life Support**
Team: Ajay & Kartik

A Mars habitat is too far from Earth for mission control to react in time when a dust storm hits. We built an autonomous controller that keeps life support running through dust storms, an aging battery, and sudden battery failures, with no human input. Its key idea: **the habitat's cabin and walls store heat, so the controller can use them as a second battery.** It pre-heats the cabin with surplus midday solar and lets it coast through the night, saving electrical energy for life support.

**Headline results (200 randomized missions, frozen settings):** compared with a rule-based priority-shedding controller, our controller was **safer in 117 missions and never less safe**. Average life-support outage fell from **17.8 h to 2.8 h**, and the worst-case outage from **136 h to 18 h**.

![Mission timeline](results/timeline.png)

---

## Problem

The habitat runs on solar panels and a battery bank. Loads fall into priority tiers:

| Tier | Load | Power | Rule |
|---|---|---|---|
| Critical | Life support (CO₂ scrubbing, O₂, water) | 3.0 kW | Never shed |
| Essential | Comms, avionics | 1.5 kW | Shed only if needed |
| Heating | Cabin heaters | 0 to 10 kW | Keep cabin 18 to 26 °C |
| Deferrable | Science, ISRU, rover charging | 4.0 kW (08:00 to 18:00) | Shed first |

Three things go wrong during a mission: a dust storm cuts solar output, the battery loses capacity as it ages, and a battery string fails suddenly. The controller must decide every hour what to power, with no human in the loop.

## Approach

### 1. Heat equation thermal model (`habitat/thermal.py`)
Heat flows from the cabin through a 10 cm insulated wall into Mars air at roughly −95 to −25 °C. We model the wall with the **1D heat equation**, discretized by finite differences (10 nodes) and integrated with backward Euler, coupled to a lumped cabin node. The result is a linear state-space model, `x[k+1] = Ad·x[k] + Bq·Q[k] + Ba·T_amb[k]`, which fits directly inside the optimizer.

**Why 1D:** the wall is about 10 cm thick and the habitat is meters across, so heat flows almost entirely through the wall's thickness (the thin-wall approximation). A 3D model would need an invented geometry, add no accuracy given uncertain parameters, and be too slow to re-solve every hour.

### 2. Fourier solar forecast and storm detection (`habitat/forecast.py`)
The forecaster fits a sum of harmonics with the Mars-day period (24.66 h) to **past measured generation only**, then extrapolates 36 hours ahead. It never sees the true storm schedule. A storm detector compares each sol's actual energy to its forecast and raises an alert when the shortfall exceeds a threshold **learned from the forecaster's own past errors** (median − 4·MAD), so there is no hand-set cutoff.

### 3. Thermal-aware model predictive control (`habitat/mpc.py`)
Every hour, the controller solves a linear program over the next 36 hours, choosing heater power, which loads run, and battery charge/discharge, subject to power balance, battery dynamics, the heat-equation model, the cabin comfort band, and a battery reserve. It applies only the first hour's decision, then re-plans (receding horizon). Priorities are encoded as penalties: losing life support costs far more than a cold cabin, which costs more than losing comms, which costs more than skipping science.

**Pre-heating emerges on its own.** Nobody programmed it: during storms the optimizer heats the cabin to about 23.5 °C at midday, then coasts down to the 18.5 °C floor overnight, because storing midday energy as heat avoids battery losses and the capacity lost in the string failure.

### 4. Baselines (`habitat/controllers.py`)
- **Naive:** runs everything, holds 21 °C.
- **Rule-based (the real baseline):** sheds science below 50% battery, lowers the heat setpoint below 30%, cuts essential systems below 15%.

## Data: real vs simulated

| Input | Source | Real? |
|---|---|---|
| Battery capacity fade | NASA Li-ion Battery Aging data (B0005, B0006, B0018), 1 cycle = 1 sol | **Real** |
| Battery end of life | 1.4 Ah (30% fade), per dataset | **Real** |
| Dust opacity range | Opportunity measured tau ≈ 0.5 normally and a record 10.8 on June 10, 2018 | **Real values**, stylized storm shape |
| Solar irradiance | Mean solar constant at Mars, 590 W/m² | **Real constant** |
| Thermal, load, array parameters | Assumptions in `habitat/config.py` | Simulated (see sensitivity analysis) |

No public dataset of real habitat life-support telemetry exists, so the habitat itself is simulated, with all assumptions in one file.

## How to run

```bash
pip install -r requirements.txt
# put the NASA battery .mat files anywhere under ./data
python run_scenario.py                     # one storm scenario, 3 controllers, timeline plot (~10 s)
python run_monte_carlo.py --n 200          # 200 randomized missions (~4 min on 8 cores)
python run_analysis.py all                 # severity, detector, sensitivity analyses
python validate_thermal.py                 # thermal model verification
```

All outputs go to `results/`.

## Results

### Single scenario: storm (tau 3.5) + 25% battery string failure on sol 14

| Controller | Life-support outage | Hours below 18 °C | Lowest cabin temp | Science done | Essential uptime |
|---|---|---|---|---|---|
| Naive | 46 h | 64 h | 12.4 °C | 98% | 95% |
| Rule-based | 4 h | 83 h | 14.7 °C | 82% | 94% |
| **MPC (ours)** | **0 h** | **0 h** | **18.5 °C** | **84%** | 85% |

The MPC is the only controller that keeps life support powered and the cabin safe for the whole mission, and it still completes more science than rule-based. The tradeoff is essential-system uptime: during the storm it deliberately turns off comms to protect life support and cabin temperature.

### 200 randomized missions, frozen settings

Each mission randomizes storm timing, peak severity (tau 2.0 to 4.5), duration, battery failure time and size (or none), which NASA cell, and starting battery age. No controller is retuned between missions.

| Controller | Zero-outage missions | Avg outage | Zero-cold missions | Avg cold hours | Science | Essential uptime |
|---|---|---|---|---|---|---|
| Naive | 50% | 39.4 h | 50% | 61.9 h | 97% | 96% |
| Rule-based | 65% | 17.8 h | 42% | 79.0 h | 86% | 94% |
| **MPC** | **68%** | **2.8 h** | **66%** | **47.5 h** | **89%** | 90% |

**Paired comparison on the same missions:** MPC safer in 117, tied in 83, **less safe in 0**.

![Monte Carlo results](results/monte_carlo.png)

### By storm severity

| Storm | Missions | Naive | Rule-based | MPC |
|---|---|---|---|---|
| Mild (tau < 3) | 85 | 100% / 0 h | 100% / 0 h | 100% / 0 h |
| Moderate (3 to 3.75) | 63 | 24% / 110 h | 70% / 33 h | **79% / 8 h** |
| Severe (> 3.75) | 52 | 0% / 179 h | 2% / 136 h | **4% / 18 h** |

*Zero-outage rate / worst-case life-support outage.*

In mild storms, every controller survives. In severe storms, no solar-only habitat collects enough energy to avoid some outage. The difference is how badly things go: **the MPC turns multi-day blackouts into outages of hours.** When energy truly runs out, it keeps life support powered and lets the cabin get cold, by design, since losing CO₂ scrubbing is worse than being cold.

### Autonomous storm detection (200 missions)

- Storms detected: **96%**, median **1.4 sols** after onset (90th percentile 2.2)
- False alarms: **4 on 921 clear sols (0.4%)**
- Fourier day-ahead forecast error on clear sols: **1.6%** of daily energy

The MPC adapts through the forecast itself; the storm flag serves as an alert to the crew.

### Robustness to modeling assumptions (30 missions per setting, no retuning)

| Setting | MPC better | MPC worse | Rule-based avg outage | MPC avg outage |
|---|---|---|---|---|
| Baseline | 20 | 0 | 19.5 h | 2.9 h |
| Thin wall (7 cm) | 30 | 0 | 61.1 h | 9.3 h |
| Thick wall (15 cm) | 6 | 0 | 0.4 h | 0.0 h |
| Half cabin heat capacity | 21 | 0 | 18.9 h | 2.9 h |
| Double cabin heat capacity | 18 | 0 | 18.5 h | 2.6 h |
| Small battery (200 kWh) | 26 | 0 | 27.7 h | 4.1 h |
| Big battery (400 kWh) | 17 | 0 | 13.6 h | 2.3 h |
| Worse insulation (k = 0.045) | 30 | 0 | 68.0 h | 10.4 h |

Across **240 missions and 8 parameter variations, the MPC was never less safe**, and it cut average outage 5 to 7 times in every setting where outages occurred. The harder the habitat, the more the controller matters.

### Thermal model verification

We checked the controller's thermal model against an **independent Crank-Nicolson heat-equation solver** written by Ajay (`heat1d.py`, second-order in time, validated against the exact sine-decay solution).

- **Wall:** after a sudden outside cold snap, the controller's model (1 h steps, 10 nodes) misses the wall's first-hour transient by up to 4.2 °C, then matches the fine solver (5 s steps, 201 nodes) to within 0.01 °C from hour 6. The steady heat flux matches exactly (34.8 W/m²).
- **Cabin:** with heaters off for 48 h, cabin temperature error versus a 36 s reference is at most **0.06 °C**.

The wall's transient lasts about 3 hours (L²/α), while the cabin responds over days (C/UA ≈ 120 h), so 1-hour steps are accurate for the quantity every control decision depends on.

![Thermal verification](results/thermal_validation.png)

### Battery remaining-life forecast (MATLAB)

TODO (Ajay): method, `capacity_fade.png`, and the predicted vs actual end-of-life table from `matlab/`.

## Limitations and next steps

- **Simulated habitat:** thermal, load, and array parameters are assumptions. The sensitivity analysis shows the conclusions hold across plausible ranges, but real habitat data would be the true test.
- **Lab battery data:** the NASA cells were cycled at controlled temperatures, not Mars cold. Mapping one lab cycle to one sol is a modeling choice.
- **Stylized storms:** storm shapes are ramps and decays using real tau values, not real tau time series.
- **Passive heat rejection:** we assume radiator louvers dump excess heat above 26 °C at no power cost.
- **Severe storms:** at 2018's record tau of 10.8, a solar-only habitat cannot survive with any controller. Real habitats would need backup power, such as a small fission unit; finding the smallest backup each controller needs is a natural next step.
- **Next:** feed a real telemetry anomaly detector (e.g., on NASA SMAP data) into the failure flags, use a mixed-integer program for on/off loads, and add chance constraints on battery uncertainty.

## Repository layout

```
habitat/
  config.py       all parameters, real values marked [REAL]
  solar.py        Mars solar + dust attenuation
  thermal.py      1D heat equation + cabin, linear state-space
  battery.py      pack driven by NASA capacity fade + string failure
  forecast.py     Fourier forecast + self-calibrating storm detector
  controllers.py  naive and rule-based baselines
  mpc.py          thermal-aware MPC (linear program, HiGHS)
  simulator.py    closed-loop simulation, metrics, plots
heat1d.py           independent Crank-Nicolson solver (verification)
run_scenario.py     single scenario
run_monte_carlo.py  randomized missions
run_analysis.py     severity, detector, sensitivity
validate_thermal.py thermal model verification
results/            all plots and CSVs
```

## Dataset credits

- **NASA Li-ion Battery Aging Data Set:** B. Saha and K. Goebel, NASA Ames Prognostics Center of Excellence (PCoE), NASA Ames Research Center.
- **2018 dust storm opacity:** NASA/JPL, "Opportunity Hunkers Down During Dust Storm" (June 2018), and JPL Photojournal PIA22930, "Last Images Opportunity Took."
