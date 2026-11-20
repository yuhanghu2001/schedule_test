// 与仿真传输模型一致：1 MB 序列化时间 + 占用产生的排队时间 + 传播时间。
export function referenceDelay(link, direction) {
  if (!link || link.status !== 'active') return null
  const availableRate = link.rate_mbps * (1 - link['background_' + direction])
  const queueMs = 20 * link['used_' + direction + '_mb'] / link['capacity_' + direction + '_mb']
  return 8000 / availableRate + queueMs + link.prop_ms
}

export const linkKey = link => link.source + '-' + link.target
export const linkName = link => 'E' + link.source + ' ↔ ' + (link.kind === 'cloud' ? 'C0' : 'E' + link.target)
