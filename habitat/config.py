"""All physical and scenario parameters in one place.

Every number here is an ASSUMPTION unless marked [REAL]. Keep the README's
real-vs-simulated table in sync with this file.
"""
from dataclasses import dataclass, field

SOL_HOURS = 24.6597  # [REAL] length of a Mars solar day in hours


@dataclass
class SolarConfig:
    array_area_m2: float = 400.0
    panel_eff: float = 0.25
    solar_const_w_m2: float = 590.0  # [REAL] mean solar irradiance at Mars orbit
    # Effective dust attenuation: P ~ exp(-k_tau * tau). A pure Beer-Lambert
    # exp(-tau/mu) overstates the loss because Mars dust scatters light diffusely.
    # k_tau = 0.3 gives ~4-5% of clear-sky output at tau ~10.8, roughly matching
    # Opportunity's reported collapse in June 2018. TODO: verify and cite.
    k_tau: float = 0.3
    noise_std: float = 0.05  # multiplicative measurement/weather noise
    baseload_kw: float = 0.0  # optional fission unit (Kilopower-style); 0 = solar only


@dataclass
class BatteryConfig:
    nominal_kwh: float = 300.0  # pack size when "new" (scaled by NASA fade curve)
    rated_ah: float = 2.0  # [REAL] rated capacity of NASA 18650 cells
    eff_charge: float = 0.95
    eff_discharge: float = 0.95
    cell_id: str = "B0005"  # [REAL] which NASA battery drives degradation
    start_cycle: int = 60  # habitat battery starts this many cycles into its life
    cycles_per_sol: float = 1.0  # one charge/discharge cycle per sol
    init_soc: float = 0.8


@dataclass
class LoadConfig:
    critical_kw: float = 3.0  # life support: CO2 scrubbing, O2, water. NEVER shed.
    essential_kw: float = 1.5  # comms, avionics
    deferrable_kw: float = 4.0  # science, ISRU, rover charging
    deferrable_hours: tuple = (8.0, 18.0)  # local-time work window
    heater_max_kw: float = 10.0
    internal_gain_frac: float = 0.6  # fraction of electrical load that ends up as cabin heat


@dataclass
class ThermalConfig:
    wall_area_m2: float = 300.0
    wall_thickness_m: float = 0.10
    wall_k: float = 0.03  # W/m/K, insulation conductivity
    wall_rho_c: float = 3.0e4  # J/m^3/K, volumetric heat capacity of insulation
    n_wall_nodes: int = 10
    h_in: float = 3.0  # W/m^2/K, cabin air <-> wall
    h_out: float = 1.5  # W/m^2/K, wall <-> thin Mars atmosphere (+ radiation, lumped)
    cabin_C: float = 3.0e7  # J/K, air + interior mass + water tanks
    T_set: float = 21.0
    T_min: float = 18.0  # safe band
    T_max: float = 26.0
    ambient_mean: float = -60.0  # deg C, equatorial Mars (assumption, cite)
    ambient_amp: float = 35.0
    ambient_min_hour: float = 5.0  # coldest just before dawn


@dataclass
class StormConfig:
    tau_clear: float = 0.5
    tau_peak: float = 3.5  # 2018 storm peaked ~10.8 at Opportunity (verify); 3.5 = hard but survivable
    start_sol: float = 8.0
    ramp_sols: float = 4.0
    plateau_sols: float = 6.0
    decay_sols: float = 20.0


@dataclass
class FailureConfig:
    string_fail_sol: float = 14.0  # sol at which a battery string dies (None to disable)
    string_fail_frac: float = 0.25  # fraction of capacity lost


@dataclass
class Scenario:
    sols: int = 40
    dt_h: float = 1.0
    seed: int = 0
    solar: SolarConfig = field(default_factory=SolarConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    loads: LoadConfig = field(default_factory=LoadConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    storm: StormConfig = field(default_factory=StormConfig)
    failure: FailureConfig = field(default_factory=FailureConfig)
    data_dir: str = "data"  # searched recursively for NASA B00xx.mat files
