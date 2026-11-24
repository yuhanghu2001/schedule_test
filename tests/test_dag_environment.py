import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from src.config import load_config
from src.environment import EdgeEnv
from src.tasks import CPU, NPU, FPGA, SubTask, TaskDAG
from src.topology import node_tier_roles


def graph(dag_id, ops=(0,), parents=((),), work=(1.,), *, slot=0, deadline=20.,
          input_mb=.5, output_mb=.25, access=0):
    """小图用给定拓扑序和关键度，方便手算派发顺序与各项物理开销。"""
    chips = ((CPU,), (CPU, FPGA), (CPU, FPGA), (CPU, NPU, FPGA), (CPU,))
    nodes = [SubTask(dag_id, i, str(i), op, chips[op], work[i], 2, .5, 1.,
                     12000, 24000, 32, 64, parents[i],
                     tuple(j for j, preds in enumerate(parents) if i in preds),
                     rank_u=float(len(ops) - i)) for i, op in enumerate(ops)]
    edges = {(p, i): .25 for i, preds in enumerate(parents) for p in preds}
    return TaskDAG(dag_id, 0, access, 0, input_mb, output_mb, sum(work), deadline, .75, 1,
                   slot, nodes, edges, (0,), (len(nodes) - 1,))


def environment(dags, **options):
    settings = dict(n_edges=3, cloud_gateway_count=1, episode_slots=2,
                    terminal_radio_rates_mbps=(160.,) * 24, service_deployment=((1,) * 5,),
                    tier_cpu_gflops=(40.,) * 4, tier_npu_gflops=(40.,) * 4,
                    tier_fpga_gflops=(40.,) * 4, link_rate_min_mbps=160., link_rate_max_mbps=160.,
                    link_rate_fluctuation=0., link_flap_prob=0., congestion_min=.25,
                    congestion_max=.25, congestion_noise=0., compute_degradation_max=0.)
    cfg = replace(load_config("hetero_dag_smoke"), **(settings | options))
    with patch("src.environment.sample_dag_tasks", return_value=dags):
        return EdgeEnv(cfg)


def row(env, node, chip, route=None):
    return next(k for k, a in enumerate(env.actions) if a.destination == node and a.resource == chip
                and (route is None or a.path == route))


class DAGEnvironmentTest(unittest.TestCase):
    def dispatch(self, env, node, chip):
        k = row(env, node, chip)
        self.assertTrue(env.action_mask[k])
        return env.step(k)

    def test_diamond_join_dma_multihop_and_delivery(self):
        diamond = graph(0, (0, 1, 1, 3), ((), (0,), (0,), (1, 2)), (1., 1., 2., 1.))
        filler = graph(1, (3,), input_mb=.75, output_mb=.5)
        env = environment([diamond, filler], episode_slots=1)
        old = env.observe()
        snapshots = {key: value.copy() for key, value in old.items() if isinstance(value, np.ndarray)}
        np.testing.assert_array_equal(old["dag_node_mask"], [1, 1, 1, 1, 0, 0])
        self.assertEqual(old["edges"].shape, (4, 4, 4))
        self.assertEqual(old["dag_edges"].shape, (6, 6, 1))
        adj = np.zeros((6, 6))
        adj[:4, :4] = np.eye(4)
        for p, i in diamond.edge_data_mb:
            adj[i, p] = 1.
            self.assertEqual(old["dag_edges"][i, p, 0], .5)
        np.testing.assert_array_equal(old["dag_adj"], adj)
        self.assertTrue(np.all(old["dag_nodes"][4:] == 0))
        self.assertTrue(old["adj"][0, env.cloud])
        self.assertFalse(old["gat_adj"][0, env.cloud])
        self.assertEqual(old["action_mask"].dtype, np.bool_)
        self.assertTrue(old["action_mask"][-1])
        self.assertFalse(env.action_mask[row(env, 0, NPU)])

        self.dispatch(env, 0, CPU)
        self.assertEqual(env.current_dag.dag_id, 1)
        self.assertEqual(env.current_time, 0.)
        pool = env.resources
        self.assertEqual(pool.free["ram_gb"][0], pool.capacity["ram_gb"][0] - .5)
        self.assertEqual(pool.free["cpu_cores"][0], pool.capacity["cpu_cores"][0] - 2)
        self.dispatch(env, 2, NPU)  # 为汇合流预先占用同向弧预算
        self.assertEqual(pool.free["npu_vram_gb"][2], pool.capacity["npu_vram_gb"][2] - 1.)
        self.assertEqual(pool.free["cpu_cores"][0], pool.capacity["cpu_cores"][0])
        self.assertEqual(env.current_subtask.subtask_id, 1)
        branch = env.actions[row(env, 0, CPU)]
        self.assertEqual(branch.transport.incoming, 0.)
        self.assertEqual(branch.transport.energy, 0.)
        self.dispatch(env, 0, CPU)
        dma = env.actions[row(env, 0, FPGA)]
        self.assertAlmostEqual(dma.transport.incoming, env.cfg.pcie_dma_base_latency_s
                               + 8 * .25 / env.cfg.pcie_dma_bandwidth_mbps)
        self.assertAlmostEqual(dma.transport.energy, .25 * env.cfg.pcie_dma_energy_per_mb_j)
        self.assertEqual(dma.flows[0].comm_tier, 1)
        self.assertEqual(env.current_time, branch.dispatch_time_s)
        obs, _, _, _ = self.dispatch(env, 0, FPGA)
        self.assertEqual(obs["current_subtask_idx"], 3)
        np.testing.assert_array_equal(obs["dag_nodes"][:4, 11:15],
                                      [[0, 0, 0, 1]] * 3 + [[0, 1, 0, 0]])
        branches = env.dag_states[0].assignments
        self.assertEqual(env.current_time, max(branches[i].action.finish_time_s for i in (1, 2)))
        self.assertNotEqual(branches[1].action.finish_time_s, branches[2].action.finish_time_s)
        self.assertNotIn((0, 2), env.active_reservations)
        self.assertLess(env.current_time, 1.)
        out = env.actions[row(env, 2, CPU)]
        self.assertEqual([f.path for f in out.flows], [(0, 1, 2), (0, 1, 2), (2, 1, 0)])
        arrival_times = []
        energy = 0.
        for pred, path, _, mb in out.flows:
            delay = 0.
            for a, b in zip(path, path[1:]):
                committed = filler.input_data_mb if pred >= 0 else filler.output_mb
                self.assertAlmostEqual(env.slot_link_used[a, b], committed)
                wire = 8 * mb / 120. + .02 * committed / 15.
                delay += wire + env.prop[a, b]
                energy += wire * env.forward_power[a, b]
            if pred >= 0:
                arrival_times.append(max(env.current_time, branches[pred].action.finish_time_s) + delay)
            else:
                down = 8 * diamond.output_mb / 160.
                return_time = delay + down
                energy += env.cfg.uplink_circuit_power_w * down
        self.assertAlmostEqual(out.start_time_s, max(arrival_times))
        self.assertAlmostEqual(out.transport.energy, energy, places=6)
        self.assertAlmostEqual(out.delivery_time_s - out.finish_time_s, return_time)
        _, _, done, _ = self.dispatch(env, 2, CPU)
        self.assertTrue(done)
        self.assertEqual(env.dag_states[0].terminal_time, out.delivery_time_s)
        self.assertTrue(env.dag_states[0].success)
        self.assertEqual(env.metrics()["comm_tier0_ratio"], .25)
        self.assertEqual(env.metrics()["comm_tier1_ratio"], .25)
        self.assertEqual(env.metrics()["comm_tier2_ratio"], .5)
        self.assertFalse(env.active_reservations)
        for key in pool.free:
            np.testing.assert_array_equal(pool.free[key], pool.capacity[key])
        for key, value in snapshots.items():
            np.testing.assert_array_equal(old[key], value)

    def test_shared_arc_budget_and_secondary_flow_route(self):
        diamond = graph(0, (0, 0, 0, 0), ((), (0,), (0,), (1, 2)), (.1,) * 4)
        diamond.edge_data_mb[1, 3] = .65
        diamond.edge_data_mb[2, 3] = .55
        # 线形网络只有一条路，环形网络可将次大流放在另一条路径。每弧每隙 1 MB。
        for ring in (False, True):
            with self.subTest(ring=ring):
                env = environment([diamond, graph(1, slot=1)], n_edges=4, undirected_edge_links=4,
                                  edge_topology="ring" if ring else "eight_neighbor",
                                  link_rate_min_mbps=32/3, link_rate_max_mbps=32/3)
                for _ in range(3):
                    self.dispatch(env, 0, CPU)
                rows = [k for k, a in enumerate(env.actions) if a.destination == 2 and a.resource == CPU]
                self.assertEqual(env.action_mask[rows].tolist(), [ring] * len(rows))
                if ring:
                    self.assertEqual(len(rows), 2)
                    for k in rows:
                        a = env.actions[k]
                        self.assertEqual(a.path, a.flows[0].path)
                        self.assertNotEqual(a.path, a.flows[1].path)
                    env.step(rows[0])
                    self.assertEqual(env.slot, 1)
                    self.assertFalse(env.slot_link_used.any())
                else:
                    self.assertTrue(env.action_mask[row(env, 0, CPU)])

    def test_fpga_context_mutex_warm_reuse_and_switch(self):
        dag = graph(0, (0, 1, 1, 2, 2), ((), (0,), (0,), (0,), (1, 2, 3)),
                    (.1, 2., 4., .2, 1.))
        env = environment([dag], episode_slots=1, compute_degradation_max=.15)
        self.dispatch(env, 0, CPU)
        pool = env.resources
        a = env.actions[row(env, 0, FPGA)]
        env.observe()
        self.assertEqual(pool.resident_op[0, FPGA], -1)
        self.assertEqual(a.setup_delay_s, env.cfg.fpga_reconfig_delay_s)
        self.assertEqual(a.setup_energy_j, env.cfg.fpga_reconfig_energy_j)
        self.dispatch(env, 0, FPGA)
        first = env.dag_states[0].assignments[1]
        eta = first.compute_degradation
        old_finish = first.action.finish_time_s
        b = env.actions[row(env, 0, FPGA)]
        self.assertEqual((b.setup_delay_s, b.setup_energy_j), (0., 0.))
        self.dispatch(env, 0, FPGA)
        self.assertEqual(first.compute_degradation, eta)
        self.assertGreater(first.action.finish_time_s, old_finish)
        self.assertAlmostEqual(first.action.finish_time_s - first.action.start_time_s,
                               (2. / 1.5) / (first.reservation.compute_f * (1 - eta)))
        self.assertGreaterEqual(b.start_time_s, first.action.start_time_s)
        self.assertEqual(pool.queues[0, FPGA], 2)
        self.assertEqual(pool.resident_op[0, FPGA], 1)
        self.assertFalse(env.action_mask[row(env, 0, FPGA)])
        for key in ("fpga_lut", "fpga_ff", "fpga_bram", "fpga_dsp", "ram_gb"):
            self.assertEqual(pool.free[key][0], pool.capacity[key][0] - 2 * getattr(dag.subtasks[1], key))
        self.assertEqual(pool.free["storage_gb"][0], pool.capacity["storage_gb"][0] - 2 * dag.storage_gb)
        self.dispatch(env, 0, CPU)
        self.assertEqual(pool.queues[0, FPGA], 0)
        self.assertEqual(pool.resident_op[0, FPGA], 1)
        switch = env.actions[row(env, 0, FPGA)]
        self.assertEqual(switch.setup_delay_s, env.cfg.fpga_reconfig_delay_s)
        self.dispatch(env, 0, FPGA)
        self.assertEqual(pool.resident_op[0, FPGA], 2)
        self.assertAlmostEqual(env.metrics()["fpga_reconfig_rate"], 2/3)
        self.assertAlmostEqual(sum(a.action.setup_energy_j for a in env.assignments),
                               2 * env.cfg.fpga_reconfig_energy_j)

    def test_same_time_kkt_and_frozen_residual_capacity_after_boundary(self):
        dags = [graph(0, work=(4.,)), graph(1, work=(16.,)),
                graph(2, slot=1), graph(3, slot=2)]
        env = environment(dags, episode_slots=3, tier_cpu_gflops=(10.,) * 4)
        self.dispatch(env, 0, CPU)
        first = env.assignments[0].action
        first_energy = first.estimated_energy
        self.assertEqual(first.compute_share, 10.)
        _, _, _, info = self.dispatch(env, 0, CPU)
        second = env.assignments[1].action
        self.assertAlmostEqual(info["reward_energy_part"], -env.cfg.reward_beta_energy
                               * (first.estimated_energy + second.estimated_energy - first_energy)
                               / env.cfg.reward_energy_ref)
        self.assertEqual(env.current_time, 1.)
        self.assertAlmostEqual(first.compute_share, 10/3)
        self.assertAlmostEqual(second.compute_share, 20/3)
        self.assertFalse(env.action_mask[row(env, 0, CPU)])
        frozen = [(a.compute_share, a.finish_time_s, a.energy) for a in (first, second)]
        self.dispatch(env, 1, CPU)
        self.assertEqual(env.current_time, 2.)
        self.assertNotIn((0, 0), env.active_reservations)
        self.assertIn((1, 0), env.active_reservations)
        new = env.actions[row(env, 0, CPU)]
        self.assertAlmostEqual(new.compute_share, 10/3)
        self.dispatch(env, 0, CPU)
        self.assertEqual(frozen, [(a.compute_share, a.finish_time_s, a.energy) for a in (first, second)])
        self.assertFalse(env.active_reservations)

    def test_boundary_samples_before_release_and_delivery_can_follow_release(self):
        # 上传恰好 1/8 秒，执行 7/8 秒，下行 1/8 秒。
        first = graph(0, work=(8.75,), input_mb=2.5, output_mb=2.5, deadline=1.125)
        env = environment([first, graph(1, slot=1)], tier_cpu_gflops=(10.,) * 4)
        self.dispatch(env, 0, CPU)
        self.assertEqual(env.current_time, 1.)
        self.assertEqual(env.assignments[0].action.finish_time_s, 1.)
        self.assertNotIn((0, 0), env.active_reservations)
        self.assertIsNone(env.dag_states[0].terminal_time)
        np.testing.assert_allclose(env.slot_load_samples[0], [np.std([1., 0., 0.]), 0., 0.])
        self.assertEqual(env.actions[row(env, 0, CPU)].compute_share, 10.)
        self.dispatch(env, 0, CPU)
        self.assertTrue(env.dag_states[0].success)
        self.assertEqual(env.dag_states[0].terminal_time, first.deadline_s)

        # 十进制工作量使精确可行的下界相减后有数个 ULP 的舍入误差。
        tight = 8 * (.1 + .25) / 160. + .01 / 40.
        env = environment([graph(0, work=(.01,), input_mb=.1, deadline=tight)])
        self.dispatch(env, 0, CPU)
        self.assertTrue(env.dag_states[0].success)

    def test_entry_peak_radio_bound_then_kkt_updates_uplink_power(self):
        tight = graph(0, deadline=.2)
        env = environment([tight, graph(1, work=(4.,))], tier_cpu_gflops=(10.,) * 4)
        self.dispatch(env, 0, CPU)
        first = env.assignments[0].action
        self.assertLess(first.uplink_power, env.cfg.uplink_max_power_w)
        self.dispatch(env, 0, CPU)
        lower = 1. / (.2 - 8 * (tight.input_data_mb + tight.output_mb) / 160.)
        self.assertAlmostEqual(first.compute_share, lower)
        self.assertAlmostEqual(env.assignments[1].action.compute_share, 10. - lower)
        self.assertAlmostEqual(first.uplink_power, env.cfg.uplink_max_power_w)
        self.assertAlmostEqual(first.delivery_time_s, .2)
        self.assertTrue(env.dag_states[0].success)

    def test_failure_freezes_committed_group_even_at_same_timestamp(self):
        first = graph(0, (0, 0, 0, 0), ((), (0,), (0,), (1, 2)), (1., 2., 1., 1.))
        other = graph(1, (0, 0), ((), (0,)), (1., 1.), access=1)
        env = environment([first, other])
        self.dispatch(env, 0, CPU)
        self.dispatch(env, 1, CPU)
        self.dispatch(env, 0, CPU)
        a = env.dag_states[0].assignments[1].action
        frozen = (a.compute_share, a.finish_time_s, a.energy)
        env.step(len(env.actions) - 1)
        self.assertEqual(env.current_dag.dag_id, 1)
        self.assertEqual(env.current_time, a.dispatch_time_s)
        self.assertFalse(env.action_mask[row(env, 0, CPU)])
        self.dispatch(env, 1, CPU)
        self.assertEqual((a.compute_share, a.finish_time_s, a.energy), frozen)

    def test_cloud_as_flow_source_and_npu_context(self):
        dag = graph(0, (0, 3, 3, 3), ((), (0,), (1,), (2,)), (1.,) * 4)
        env = environment([dag])
        self.dispatch(env, env.cloud, CPU)
        a = env.actions[row(env, 2, NPU)]
        self.assertEqual(a.flows[0].path, (env.cloud, 0, 1, 2))
        self.assertEqual(a.setup_delay_s, env.cfg.npu_context_load_delay_s)
        self.assertEqual(a.setup_energy_j, env.cfg.npu_context_load_energy_j)
        self.dispatch(env, 2, NPU)
        self.assertEqual(env.resources.resident_op[2, NPU], dag.subtasks[1].op_type)
        warm = env.actions[row(env, 2, NPU)]
        self.assertEqual((warm.setup_delay_s, warm.setup_energy_j), (0., 0.))
        self.dispatch(env, 2, NPU)
        out = env.actions[row(env, env.cloud, CPU)]
        self.assertEqual(out.flows[0].path, (2, 1, 0, env.cloud))
        self.assertEqual(out.flows[-1].path, (env.cloud, 0))
        self.assertTrue(all(env.cloud not in f.path[1:-1] for a in env.actions for f in a.flows))
        self.dispatch(env, env.cloud, CPU)
        self.assertTrue(env.dag_states[0].success)

    def test_mid_graph_failure_cancels_pending_and_charges_once(self):
        for choice in ("reject", "masked", "out_of_bounds"):
            with self.subTest(choice=choice):
                dag = graph(0, (0, 0, 0, 0), ((), (0,), (0,), (1, 2)), (.1, 60., 1., 1.))
                forced = graph(1, slot=1)
                forced.subtasks[0].ram_gb = 1000.
                env = environment([dag, forced])
                self.dispatch(env, 0, CPU)
                self.dispatch(env, 0, CPU)
                spent = sum(a.action.energy for a in env.assignments)
                finish = env.assignments[-1].action.finish_time_s
                fail_at = env.current_time
                k = {"reject": len(env.actions) - 1, "masked": row(env, 0, NPU),
                     "out_of_bounds": len(env.actions)}[choice]
                obs, _, done, info = env.step(k)
                self.assertFalse(done)
                self.assertEqual(info["invalid_action"], choice != "reject")
                c = env.cfg
                self.assertEqual(-info["reward_constraint_part"], c.reward_penalty_incomplete
                                 + (c.reward_penalty_conditional_reject if choice == "reject"
                                    else c.reward_penalty_invalid))
                state = env.dag_states[0]
                self.assertEqual(state.terminal_time, fail_at)
                self.assertEqual(state.status, {0: "completed", 1: "running", 2: "cancelled", 3: "cancelled"})
                self.assertEqual(env.active_reservations[0, 1].release_time, finish)
                self.assertEqual(env.metrics()["total_energy_j"], spent)
                self.assertFalse(obs["action_mask"][:-1].any())
                self.assertTrue(obs["action_mask"][-1])
                self.assertTrue(all(np.isfinite(v).all() for v in obs.values() if isinstance(v, np.ndarray)))
                _, reward, done, info = env.step(len(env.actions) - 1)
                self.assertTrue(done)
                self.assertTrue(np.isfinite(reward))
                self.assertEqual(info["reward_constraint_part"], 0.)
                self.assertEqual(env.dag_states[1].balance, (0., 0., 0.))
                self.assertEqual(state.terminal_time, fail_at)
                self.assertEqual(env.missed_dags, 1)
                self.assertFalse(env.active_reservations)
                self.assertAlmostEqual(env.metrics()["acceptance_rate"], .5)
                self.assertIsNone(env.metrics()["avg_dag_energy_j"])

    def test_reward_telescopes_against_independent_terminal_and_load_accounting(self):
        # 两图同刻 KKT 修正、主动中途拒绝、无解拒绝、非法动作、接收后超期。
        dags = [graph(0, work=(4.,)), graph(1, work=(16.,)),
                graph(2, (0, 0), ((), (0,)), (.1, 1.), access=1),
                graph(3, slot=1), graph(4, slot=1),
                graph(5, slot=1, access=2, work=(10.,), deadline=1.037501)]
        dags[3].subtasks[0].cpu_cores = 20000
        env = environment(dags, episode_slots=3, tier_cpu_gflops=(10.,) * 4,
                          compute_degradation_max=.3, reward_load_std_ref_v9=.3,
                          reward_energy_ref=5., reward_delay_ref=2.,
                          load_balance_chip_weights=(.5, .25, .25))
        rewards = []
        while True:
            dag_id = env.current_dag.dag_id
            if dag_id == 4:
                k = -1
            elif dag_id == 3 or (dag_id == 2 and env.current_subtask.subtask_id == 1):
                k = len(env.actions) - 1
            else:
                k = row(env, env.current_dag.access_edge, CPU)
                self.assertTrue(env.action_mask[k])
            _, reward, done, _ = env.step(k)
            rewards.append(reward)
            if done:
                break
        c = env.cfg
        charges = 2 * c.reward_penalty_incomplete + c.reward_penalty_invalid + c.reward_penalty_conditional_reject
        expected = -charges
        for dag, success in zip(dags, (True, True, False, False, False, False)):
            placed = [a.action for a in env.assignments if a.action.dag_id == dag.dag_id]
            if not placed:
                continue
            duration = max(a.delivery_time_s for a in placed) - dag.slot * c.slot_seconds
            # 图 2 在入口完成、唯一后继就绪的时刻被拒。
            self.assertEqual(env.dag_states[dag.dag_id].success, success)
            expected += (c.reward_beta_completion * success - c.reward_beta_delay * duration / c.reward_delay_ref
                         - c.reward_beta_energy * sum(a.energy for a in placed) / c.reward_energy_ref)
        samples = []
        for s in range(1, c.episode_slots + 1):
            loads = np.zeros((c.n_edges, 3))
            for assignment in env.assignments:
                a = assignment.action
                if a.destination < c.n_edges and a.dispatch_time_s < s * c.slot_seconds <= a.finish_time_s:
                    loads[a.destination, a.resource] += a.compute_share / env.speed[a.destination, a.resource]
            samples.append(loads.std(axis=0, ddof=0))
        self.assertGreater(np.sum(samples), 0.)
        np.testing.assert_allclose(env.slot_load_samples, samples, atol=1e-14)
        expected -= c.reward_beta_load * np.sum(np.array(samples) @ c.load_balance_chip_weights) / c.reward_load_std_ref_v9
        self.assertAlmostEqual(sum(rewards), expected, places=11)
        self.assertAlmostEqual(env.metrics()["episode_reward"], expected, places=11)

    def test_tier_capacities_overrides_and_resource_masks(self):
        cfg = load_config("hetero_dag_smoke")
        env = EdgeEnv(cfg)
        pool = env.resources
        self.assertEqual(pool.speed.shape, (13, 3))
        tiers = ("access", "middle", "gateway", "cloud")
        for m, role in enumerate((*node_tier_roles(12), "cloud")):
            i = tiers.index(role)
            np.testing.assert_array_equal(pool.speed[m], [cfg.tier_cpu_gflops[i],
                                                        cfg.tier_npu_gflops[i], cfg.tier_fpga_gflops[i]])
            self.assertEqual(pool.capacity["ram_gb"][m], cfg.tier_ram_gb[i])
        dag = graph(0, (3,))
        for key, chip in (("cpu_cores", CPU), ("npu_vram_gb", NPU), ("fpga_lut", FPGA),
                          ("fpga_ff", FPGA), ("fpga_bram", FPGA), ("fpga_dsp", FPGA)):
            with self.subTest(key=key):
                env = environment([dag], **{"tier_" + key: (0.,) * 4})
                self.assertFalse(env.action_mask[row(env, 0, chip)])
                self.assertTrue(env.action_mask[row(env, 0, NPU if chip == CPU else CPU)])
        env = environment([dag], node_memory_gb=(0., 2., 2., 2.), node_storage_gb=(1., 0., 1., 1.),
                          node_queue_capacity=(2, 2, 0, 2))
        for node in range(3):
            self.assertFalse(env.action_mask[row(env, node, CPU)])
        self.assertTrue(env.action_mask[row(env, env.cloud, CPU)])
        env = environment([dag], service_deployment=((0,) * 5,))
        self.assertFalse(env.action_mask[row(env, 0, CPU)])
        self.assertTrue(env.action_mask[row(env, env.cloud, CPU)])
