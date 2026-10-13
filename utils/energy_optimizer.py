"""
SolarSat AI - Satellite Energy Intelligence
Energy Allocation Optimization Module
Implements aerospace priority-based power dispatch simulator for critical and optional satellite loads.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def run_power_optimization(
    available_energy_Wh: float,
    critical_demand_W: float,
    optional_demand_W: float,
    battery_reserve_pct: float = 20.0,
    time_window_hours: float = 1.0,
    priority_mode: str = "Critical First (Standard Orbit)"
) -> Dict[str, Any]:
    """
    Simulates priority-based satellite energy dispatch.
    
    Rules:
    1. Usable energy is bounded by the battery reserve percentage.
    2. Usable Wh = available_energy_Wh * (1 - battery_reserve_pct / 100).
    3. Usable continuous Power capacity = Usable Wh / time_window_hours.
    4. Critical loads (OBC, ADCS, beacon) are served first.
    5. Remaining power capacity is dispatched to optional loads (payloads, imagers).
    6. Returns baseline vs optimized allocations and deficit/surplus metrics.
    """
    total_baseline_demand_W = critical_demand_W + optional_demand_W
    baseline_energy_consumed_Wh = total_baseline_demand_W * time_window_hours

    # Reserve calculation
    reserve_energy_Wh = available_energy_Wh * (battery_reserve_pct / 100.0)
    usable_energy_Wh = max(0.0, available_energy_Wh - reserve_energy_Wh)
    usable_power_budget_W = usable_energy_Wh / time_window_hours

    if priority_mode == "Critical First (Standard Orbit)":
        # 1. Allocate to critical load
        allocated_critical_W = min(critical_demand_W, usable_power_budget_W)
        unmet_critical_W = max(0.0, critical_demand_W - allocated_critical_W)

        # 2. Remaining power goes to optional load
        remaining_power_budget_W = max(0.0, usable_power_budget_W - allocated_critical_W)
        allocated_optional_W = min(optional_demand_W, remaining_power_budget_W)
        unmet_optional_W = max(0.0, optional_demand_W - allocated_optional_W)

    elif priority_mode == "Balanced Shedding":
        # Proportionally scale if usable budget is below total demand
        if usable_power_budget_W >= total_baseline_demand_W:
            allocated_critical_W = critical_demand_W
            allocated_optional_W = optional_demand_W
            unmet_critical_W = 0.0
            unmet_optional_W = 0.0
        else:
            # Shield critical to 90% minimum if possible
            allocated_critical_W = min(critical_demand_W, usable_power_budget_W * 0.8)
            unmet_critical_W = max(0.0, critical_demand_W - allocated_critical_W)
            remaining_for_opt = max(0.0, usable_power_budget_W - allocated_critical_W)
            allocated_optional_W = min(optional_demand_W, remaining_for_opt)
            unmet_optional_W = max(0.0, optional_demand_W - allocated_optional_W)
    else:
        # Strict Emergency Safe Mode: cut 100% of optional load
        allocated_critical_W = min(critical_demand_W, usable_power_budget_W)
        unmet_critical_W = max(0.0, critical_demand_W - allocated_critical_W)
        allocated_optional_W = 0.0
        unmet_optional_W = optional_demand_W

    total_optimized_power_W = allocated_critical_W + allocated_optional_W
    optimized_energy_consumed_Wh = total_optimized_power_W * time_window_hours

    energy_saved_Wh = max(0.0, baseline_energy_consumed_Wh - optimized_energy_consumed_Wh)
    energy_saving_percent = (
        (energy_saved_Wh / baseline_energy_consumed_Wh * 100.0)
        if baseline_energy_consumed_Wh > 0 else 0.0
    )

    critical_coverage_pct = (
        (allocated_critical_W / critical_demand_W * 100.0)
        if critical_demand_W > 0 else 100.0
    )
    optional_coverage_pct = (
        (allocated_optional_W / optional_demand_W * 100.0)
        if optional_demand_W > 0 else 100.0
    )

    # Surplus or deficit in Wh
    surplus_deficit_Wh = usable_energy_Wh - optimized_energy_consumed_Wh

    # Operational status
    if unmet_critical_W > 0:
        status = "CRITICAL POWER DEFICIT: Essential Loads Unmet"
        status_color = "red"
    elif unmet_optional_W > 0:
        status = "PARTIAL SHEDDING: Critical Satisfied, Optional Throttled"
        status_color = "orange"
    else:
        status = "NOMINAL: All Subsystems Fully Powered"
        status_color = "green"

    return {
        "baseline_demand_W": round(total_baseline_demand_W, 3),
        "critical_demand_W": round(critical_demand_W, 3),
        "optional_demand_W": round(optional_demand_W, 3),
        "usable_energy_Wh": round(usable_energy_Wh, 3),
        "reserve_energy_Wh": round(reserve_energy_Wh, 3),
        "allocated_critical_W": round(allocated_critical_W, 3),
        "allocated_optional_W": round(allocated_optional_W, 3),
        "unmet_critical_W": round(unmet_critical_W, 3),
        "unmet_optional_W": round(unmet_optional_W, 3),
        "total_optimized_power_W": round(total_optimized_power_W, 3),
        "baseline_energy_Wh": round(baseline_energy_consumed_Wh, 3),
        "optimized_energy_Wh": round(optimized_energy_consumed_Wh, 3),
        "energy_saved_Wh": round(energy_saved_Wh, 3),
        "energy_saving_percent": round(energy_saving_percent, 2),
        "critical_coverage_pct": round(critical_coverage_pct, 1),
        "optional_coverage_pct": round(optional_coverage_pct, 1),
        "surplus_deficit_Wh": round(surplus_deficit_Wh, 3),
        "status": status,
        "status_color": status_color
    }


def compare_precomputed_dataset_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the dataset's precomputed simulated optimization statistics across recording sessions.
    """
    opt_cols = [
        "total_power_demand_W_simulated", "critical_load_W_simulated",
        "optional_load_W_simulated", "optimized_critical_load_W_simulated",
        "optimized_optional_load_W_simulated", "energy_saving_percent_simulated"
    ]
    present_cols = [c for c in opt_cols if c in df.columns]
    if not present_cols:
        return pd.DataFrame()

    summary = df.groupby("Recording_Session")[present_cols].mean().round(3)
    return summary
