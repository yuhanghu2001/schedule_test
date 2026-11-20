// 与仿真传输模型一致：1 MB 序列化时间 + 占用产生的排队时间 + 传播时间。
export function referenceDelay(link, direction) {
  if (!link || link.status !== 'active') return null
  const availableRate = link.rate_mbps * (1 - link['background_' + direction])
  const queueMs = 20 * link['used_' + direction + '_mb'] / link['capacity_' + direction + '_mb']
  return 8000 / availableRate + queueMs + link.prop_ms
}

export const linkKey = link => link.source + '-' + link.target
export const linkName = link => 'E' + link.source + ' ↔ ' + (link.kind === 'cloud' ? 'C0' : 'E' + link.target)

export function linkHealth(links) {
  const active = links.filter(link => link.status === 'active')
  const congested = active.filter(link => Math.max(link.background_fwd, link.background_rev) >= 0.7)
    .sort((a, b) => Math.max(b.background_fwd, b.background_rev) - Math.max(a.background_fwd, a.background_rev))
  const offline = links.filter(link => link.status === 'flapped')
  const used = active.reduce((sum, link) => sum + link.used_fwd_mb + link.used_rev_mb, 0)
  const capacity = active.reduce((sum, link) => sum + link.capacity_fwd_mb + link.capacity_rev_mb, 0)
  const delay = active.reduce((sum, link) => sum + referenceDelay(link, 'fwd') + referenceDelay(link, 'rev'), 0)
  return {
    online: active.length,
    total: links.length,
    delay: active.length ? delay / (2 * active.length) : null,
    utilization: capacity ? 100 * used / capacity : null,
    congested,
    offline,
  }
}
