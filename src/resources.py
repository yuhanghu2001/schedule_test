"""DAG 模式的节点容量和半开区间预留。"""
from dataclasses import dataclass

import numpy as np

from .config import NODE_TIERS
from .tasks import CPU, FPGA, NPU
from .topology import node_tier_roles


DEMAND_FIELDS = ("cpu_cores", "ram_gb", "npu_vram_gb", "fpga_lut",
                 "fpga_ff", "fpga_bram", "fpga_dsp")


@dataclass
class ActiveReservation:
    dag_id: int
    subtask_id: int
    node: int
    chip: int
    op_type: int
    release_time: float
    compute_f: float
    cpu_cores: int
    ram_gb: float
    storage_gb: float
    npu_vram_gb: float
    fpga_lut: int
    fpga_ff: int
    fpga_bram: int
    fpga_dsp: int

    @classmethod
    def for_subtask(cls, task, node, chip, storage):
        return cls(task.dag_id, task.subtask_id, node, chip, task.op_type, 0., 0.,
                   task.cpu_cores if chip == CPU else 0, task.ram_gb, storage,
                   task.npu_vram_gb if chip == NPU else 0.,
                   task.fpga_lut if chip == FPGA else 0, task.fpga_ff if chip == FPGA else 0,
                   task.fpga_bram if chip == FPGA else 0, task.fpga_dsp if chip == FPGA else 0)


class ResourcePool:
    """算力、队列及驻留矩阵按节点排列，三列依次对应 CPU、NPU、FPGA。"""

    def __init__(self, cfg):
        tiers = [NODE_TIERS.index(role) for role in (*node_tier_roles(cfg.n_edges), "cloud")]
        self.capacity = {key: np.asarray(getattr(cfg, "tier_" + key), dtype=float)[tiers]
                         for key in DEMAND_FIELDS}
        if cfg.node_memory_gb:
            self.capacity["ram_gb"] = np.array(cfg.node_memory_gb, dtype=float)
        self.capacity["storage_gb"] = np.array(
            cfg.node_storage_gb or [1000.] * cfg.n_edges + [4000.], dtype=float)
        self.queue_cap = np.array(cfg.node_queue_capacity or [60] * cfg.n_edges + [80])
        self.speed = np.array([cfg.tier_cpu_gflops, cfg.tier_npu_gflops,
                               cfg.tier_fpga_gflops], dtype=float)[:, tiers].T
        self.resident_op = np.full(self.speed.shape, -1, dtype=int)
        self.context_ready_at = np.zeros_like(self.speed)
        self.active: dict[tuple[int, int], ActiveReservation] = {}
        self.rebuild()

    def rebuild(self):
        """从唯一预留表重算余量，同刻重分配或释放最后一份份额后无累计舍入误差。"""
        self.free = {k: cap.copy() for k, cap in self.capacity.items()}
        self.compute_used = np.zeros_like(self.speed)
        self.queues = np.zeros(self.speed.shape, dtype=int)
        for r in self.active.values():
            self.compute_used[r.node, r.chip] += r.compute_f
            self.queues[r.node, r.chip] += 1
            for key, free in self.free.items():
                free[r.node] -= getattr(r, key)
        self.free_compute = self.speed - self.compute_used

    def can_allocate(self, r):
        node, chip = r.node, r.chip
        return (self.queues[node].sum() < self.queue_cap[node]
                and (chip == CPU or not self.queues[node, chip]
                     or self.resident_op[node, chip] == r.op_type)
                and all(getattr(r, key) <= free[node] for key, free in self.free.items()))

    def load_std(self):
        fractions = np.divide(self.compute_used[:-1], self.speed[:-1],
                              out=np.zeros_like(self.speed[:-1]), where=self.speed[:-1] > 0)
        return fractions.std(axis=0)
