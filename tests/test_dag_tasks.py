import unittest
from dataclasses import replace
from graphlib import TopologicalSorter

import numpy as np

from src.config import load_config
from src.tasks import _critical_path_work, compute_dag_priorities, sample_dag_tasks
from src.topology import node_tier_roles


class DAGTaskTest(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(load_config("hetero_dag_smoke"), deadline_slack_min=2.,
                           deadline_slack_max=2., arrival_rate=12.)
        self.rates = np.linspace(60., 180., self.cfg.n_terminals)
        self.dags = self.sample(self.rates)

    def sample(self, rates):
        return sample_dag_tasks(np.random.default_rng(42), self.cfg, rates, 120.)

    def test_templates_and_dependencies(self):
        sizes = (4, 5, 6, 5, 4)
        # 每行按拓扑序给出所有实际入边及其占整图输入量的比例。
        flows = (
            [(0, 1, .5), (0, 2, .5), (1, 3, .15), (2, 3, .15)],
            [(0, 1, .5), (0, 2, .5), (1, 3, .08), (2, 3, .16), (3, 4, .02)],
            [(0, 1, .3), (0, 2, .3), (0, 3, .3), (1, 4, .06), (2, 4, .05), (3, 4, .10), (4, 5, .02)],
            [(0, 1, .5), (0, 2, .5), (1, 3, .12), (2, 3, .10), (3, 4, .02)],
            [(0, 1, .5), (0, 2, .5), (1, 3, .10), (2, 3, .08)],
        )
        self.assertEqual({d.kind for d in self.dags}, set(range(5)))
        for dag in self.dags:
            nodes = {t.subtask_id: t for t in dag.subtasks}
            order = tuple(TopologicalSorter({i: t.preds for i, t in nodes.items()}).static_order())
            self.assertEqual(len(order), sizes[dag.kind])
            self.assertEqual(tuple(i for i in order if not nodes[i].preds), dag.entry_ids)
            self.assertEqual(tuple(i for i in order if not nodes[i].succs), dag.exit_ids)
            self.assertEqual((len(dag.entry_ids), len(dag.exit_ids)), (1, 1))
            self.assertAlmostEqual(sum(t.compute_gflop for t in dag.subtasks), dag.compute_gflop)
            self.assertEqual(dag.edge_data_mb, {(p, i): frac * dag.input_data_mb
                                                for p, i, frac in flows[dag.kind]})
            self.assertEqual({(p, i) for i, t in nodes.items() for p in t.preds}, set(dag.edge_data_mb))
            self.assertEqual({(i, j) for i, t in nodes.items() for j in t.succs}, set(dag.edge_data_mb))
            for i, task in nodes.items():
                self.assertTrue(task.compatible_chips)
                self.assertTrue(all(self.cfg.operator_affinity_matrix[task.op_type][u] > 0
                                    for u in task.compatible_chips))
                for j in task.succs:
                    self.assertGreater(task.rank_u, nodes[j].rank_u)

    def test_priorities_with_parallel_branches(self):
        dag = next(d for d in self.dags if d.kind == 0)
        entry, left, right, exit_node = dag.subtasks
        for task, work in zip(dag.subtasks, (2., 6., 2., 4.)):
            task.compute_gflop = work
        dag.edge_data_mb = dict.fromkeys(dag.edge_data_mb, 1.)
        dag.subtasks.reverse()  # 计算顺序由依赖决定
        affinity = np.ones((5, 3))
        affinity[left.op_type, 2] = 2.
        compute_dag_priorities(dag, (1., 1., 1.), 8., affinity)
        # 两条分支平均执行时间分别为 (6+3)/2 和 (2+1)/2，依赖传输各 1 秒。
        self.assertAlmostEqual(entry.rank_u, 12.5)
        self.assertAlmostEqual(exit_node.est_s, 8.5)
        self.assertAlmostEqual(right.est_s, 3.)
        self.assertAlmostEqual(right.lst_s, 6.)
        self.assertAlmostEqual(right.slack_s, 3.)
        for critical in (entry, left, exit_node):
            self.assertAlmostEqual(critical.slack_s, 0.)

    def test_deadline_uses_each_sources_radio_rate_and_longest_path(self):
        slow_rates = self.rates.copy()
        source = self.dags[0].source
        slow_rates[source] /= 4
        for dag, slow in zip(self.dags, self.sample(slow_rates), strict=True):
            self.assertEqual((dag.source, dag.compute_gflop), (slow.source, slow.compute_gflop))
            # 五类模板都是一次分岔再汇合；只从各并行分支中取最大工作量。
            branches = dag.subtasks[0].succs
            work = sum(t.compute_gflop for t in dag.subtasks if t.subtask_id not in branches)
            work += max(dag.subtasks[i].compute_gflop for i in branches)
            radio_time = 8 * (dag.input_data_mb + dag.output_mb) / self.rates[dag.source]
            self.assertAlmostEqual(dag.deadline_s, 2 * (radio_time + work / 120.))
            if dag.source == source:
                self.assertGreater(slow.deadline_s, dag.deadline_s)
            else:
                self.assertEqual(slow.deadline_s, dag.deadline_s)

    def test_changing_noncritical_work_does_not_change_reference_time(self):
        dag = next(d for d in self.dags if d.kind == 1)
        short, long = dag.subtasks[1:3]
        self.assertLess(short.compute_gflop, long.compute_gflop)
        before = _critical_path_work(dag.subtasks) / 120.
        short.compute_gflop = (short.compute_gflop + long.compute_gflop) / 2
        self.assertEqual(_critical_path_work(dag.subtasks) / 120., before)

    def test_tier_roles(self):
        roles = node_tier_roles(12)
        self.assertEqual({i for i, r in enumerate(roles) if r == "gateway"}, {0, 1, 2, 3})
        self.assertEqual({i for i, r in enumerate(roles) if r == "middle"}, {11, 8, 7, 4})
        self.assertEqual({i for i, r in enumerate(roles) if r == "access"}, {10, 9, 6, 5})
        self.assertEqual(node_tier_roles(4), ("gateway",) * 4)

    def test_dag_presets_only_switch_mode_and_versions(self):
        for base, dag in (("default", "hetero_dag_default"),
                          ("high_congestion", "hetero_dag_high_congestion"),
                          ("small_smoke", "hetero_dag_smoke")):
            cfg = load_config(base)
            self.assertFalse(cfg.enable_dag_mode)
            self.assertEqual((cfg.observation_version, cfg.reward_version),
                             ("v4_edge_only_4d_edges", "v8_dynamic_bottleneck"))
            self.assertEqual(load_config(dag), replace(cfg, enable_dag_mode=True,
                                                       observation_version="v5_dual_graph_dag",
                                                       reward_version="v9_hetero_dag"))
