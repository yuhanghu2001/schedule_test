export const dispatchAlgorithms = [
  { id: 'gatppo', name: 'GAT-PPO', color: '#52c7ff' },
  { id: 'ppo', name: 'PPO', color: '#51dfb3' },
  { id: 'shortest_fastest', name: '最短最快启发式', color: '#ffbd63' },
  { id: 'greedy_estimate', name: '贪心估计启发式', color: '#b6a3ff' },
  { id: 'random', name: '随机算法', color: '#b7cee2' },
]

export const dispatchCheckpointNames = {
  gatppo: ['best_gatppo.pt', 'best_gat_ppo.pt', 'best_gat.pt'],
  ppo: ['best_ppo.pt', 'best_flat_ppo.pt', 'best_flat.pt'],
}

export function checkpointBatch(runDir) {
  const batch = runDir.split('/')[0]
  return batch.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/, '$1-$2-$3 $4:$5:$6')
}
