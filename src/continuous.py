"""连续优化子问题：KKT 并发算力分配与 Dinkelbach 上行能效优化。"""
from dataclasses import dataclass

import numpy as np


@dataclass
class UplinkSolution:
    power_w: float
    rate_mbps: float
    efficiency_mbit_per_j: float
    residual: float
    iterations: int


def allocate_compute(
    compute_gflops,
    priorities,
    capacity_gflops,
    minimum_gflops,
    priority_exponent=1.0,
):
    """解带每任务最低算力约束的加权完成时间凸子问题。"""
    compute = np.asarray(compute_gflops, dtype=np.float64)
    weights = np.asarray(priorities, dtype=np.float64) ** priority_exponent
    lower = np.asarray(minimum_gflops, dtype=np.float64)
    if lower.sum() > capacity_gflops:
        return None
    roots = np.sqrt(weights * compute)
    allocation = np.zeros_like(roots)
    free = np.ones(len(roots), dtype=bool)
    remaining = capacity_gflops
    while np.any(free):
        proposed = remaining * roots[free] / roots[free].sum()
        constrained = proposed < lower[free]
        if not np.any(constrained):
            allocation[free] = proposed
            break
        indices = np.flatnonzero(free)
        fixed = indices[constrained]
        allocation[fixed] = lower[fixed]
        remaining -= lower[fixed].sum()
        free[fixed] = False
    return allocation


def uplink_rate(power_w, bandwidth_mhz, channel_gain_to_noise):
    """Shannon 上行速率，单位 Mbps。"""
    return bandwidth_mhz * np.log2(1.0 + channel_gain_to_noise * power_w)


def choose_uplink_power(
    bandwidth_mhz,
    channel_gain_to_noise,
    circuit_power_w,
    max_power_w,
    min_rate_mbps,
    tolerance,
    max_iterations,
):
    """用 Dinkelbach 法最大化 R(p)/(p+p_c)，并满足最低速率。"""
    min_power = (2.0 ** (min_rate_mbps / bandwidth_mhz) - 1.0) / channel_gain_to_noise
    quotient = 0.0
    power = max_power_w
    for iteration in range(1, max_iterations + 1):
        if quotient > 0.0:
            stationary = bandwidth_mhz / (quotient * np.log(2.0)) - 1.0 / channel_gain_to_noise
            power = float(np.clip(stationary, min_power, max_power_w))
        rate = float(uplink_rate(power, bandwidth_mhz, channel_gain_to_noise))
        residual = rate - quotient * (power + circuit_power_w)
        if abs(residual) <= tolerance:
            break
        quotient = rate / (power + circuit_power_w)
    return UplinkSolution(
        power_w=power,
        rate_mbps=rate,
        efficiency_mbit_per_j=rate / (power + circuit_power_w),
        residual=residual,
        iterations=iteration,
    )
