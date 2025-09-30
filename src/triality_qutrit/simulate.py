from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class HotspotCfg:
    base_T2star_ns: float = 120.0
    noise_sigma_ns: float = 8.0
    center_dBm: tuple = (-46.0, -43.0)
    amplitude_ns: float = 80.0
    width_dB: tuple = (5.5, 6.0)
    detuning_linewidth_MHz: float = 2.5 # Linewidth for detuning effect

def gaussian2d(x, y, x0, y0, sx, sy):
    return np.exp(-0.5*(((x-x0)/sx)**2 + ((y-y0)/sy)**2))

def lorentzian(x, x0, fwhm):
    """A Lorentzian lineshape function."""
    gamma = fwhm / 2.0
    return (gamma**2) / ((x - x0)**2 + gamma**2)

def synthesize(csv_out: str | Path, steps=41, P_ge_range=(-60.0,-10.0), P_ef_range=(-60.0,-10.0), cfg: HotspotCfg = HotspotCfg(), randomize_phases: bool = False, pump_detuning_MHz: float = 0.0):
    P_ge_vals = np.linspace(P_ge_range[0], P_ge_range[1], steps)
    P_ef_vals = np.linspace(P_ef_range[0], P_ef_range[1], steps)
    ge_grid, ef_grid = np.meshgrid(P_ge_vals, P_ef_vals, indexing='ij')
    
    # Calculate the detuning suppression factor
    detuning_suppression = lorentzian(pump_detuning_MHz, 0, cfg.detuning_linewidth_MHz)

    if not randomize_phases:
        # The hotspot amplitude is now scaled by the detuning suppression
        hotspot_amp = cfg.amplitude_ns * detuning_suppression
        hotspot = hotspot_amp * gaussian2d(ge_grid, ef_grid, cfg.center_dBm[0], cfg.center_dBm[1], cfg.width_dB[0], cfg.width_dB[1])
    else:
        # If phases are random, the coherent hotspot should not form.
        hotspot = np.zeros_like(ge_grid)

    anis = 6.0*np.sin((ge_grid+50)/7.0) + 4.0*np.cos((ef_grid+47)/5.5) - 3.0*np.exp(-((ge_grid-ef_grid)/12.0)**2)
    T2star = cfg.base_T2star_ns + hotspot + anis + np.random.normal(0, cfg.noise_sigma_ns, size=hotspot.shape)
    rows = []
    rid = 0
    rng = np.random.default_rng()

    f_ge = 5.2e9
    f_ef = 4.9e9
    f_gf_on_resonance = f_ge + f_ef
    f_gf = f_gf_on_resonance + (pump_detuning_MHz * 1e6)


    for i, P_ge in enumerate(P_ge_vals):
        for j, P_ef in enumerate(P_ef_vals):
            rid += 1
            if randomize_phases:
                phi_ge = rng.uniform(0, 2 * np.pi)
                phi_ef = rng.uniform(0, 2 * np.pi)
                phi_gf = rng.uniform(0, 2 * np.pi)
            else:
                phi_ge, phi_ef, phi_gf = 0.0, 0.0, 0.0

            notes = []
            if randomize_phases:
                notes.append("Randomized Phases")
            if pump_detuning_MHz != 0:
                notes.append(f"Pump Detuning: {pump_detuning_MHz} MHz")


            rows.append(dict(
                run_id=f"sim_{rid:05d}",
                batch_id="sim_batch_detuned" if pump_detuning_MHz != 0 else "sim_batch",
                timestamp="",
                f_ge_Hz=f_ge, f_ef_Hz=f_ef, f_gf_Hz=f_gf,
                P_pump_dBm=0.0, P_ge_dBm=float(P_ge), P_ef_dBm=float(P_ef),
                phi_ge_rad=phi_ge, phi_ef_rad=phi_ef, phi_gf_rad=phi_gf,
                T2star_ns=float(max(10.0, T2star[i,j])),
                T1_ns=300.0, contrast=0.7, fit_ok=True,
                temp_mK=15.0, fridge_cycle_id="sim", resonator_Q=8000.0, notes="; ".join(notes)
            ))
    df = pd.DataFrame(rows)
    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    return df