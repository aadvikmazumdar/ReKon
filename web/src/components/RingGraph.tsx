import { useMemo } from "react";
import type { Application, Dataset } from "../types";

interface Props {
  data: Dataset;
  clusterId: number;
  selected?: string;
  onSelect: (id: string) => void;
}

const W = 620, H = 420, PAD = 96;

export default function RingGraph({ data, clusterId, selected, onSelect }: Props) {
  const cluster = data.clusters[String(clusterId)];

  const layout = useMemo(() => {
    if (!cluster) return null;
    const members = cluster.members
      .map((id) => data.applications.find((a) => a.id === id))
      .filter((a): a is Application => Boolean(a))
      .sort((a, b) => a.date.localeCompare(b.date));
    const n = members.length;
    const cx = W / 2, cy = H / 2, r = Math.min(W, H) / 2 - PAD;
    const pos = new Map<string, { x: number; y: number }>();
    members.forEach((m, i) => {
      const a = (2 * Math.PI * i) / n - Math.PI / 2;
      pos.set(m.id, n === 1
        ? { x: cx, y: cy }
        : { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
    });
    return { members, pos };
  }, [cluster, data.applications]);

  if (!cluster || !layout || layout.members.length < 2) {
    return <p className="note">This application is not linked to any earlier record.</p>;
  }

  const { members, pos } = layout;
  const flagged = members.filter(
    (m) => m.status === "rejected" || m.status === "blocklisted",
  ).length;

  return (
    <div className="ring-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img"
           aria-label={`Link graph of ${members.length} applications`}>
        {cluster.edges.map((e, i) => {
          const a = pos.get(e.a), b = pos.get(e.b);
          if (!a || !b) return null;
          return (
            <g key={i}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke="#c3cbd4" strokeWidth={1 + Math.min(3, Math.abs(e.score) / 40)} />
              <text x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 4}
                    textAnchor="middle" fontFamily="IBM Plex Mono, monospace"
                    fontSize="10" fill="#8b97a3">{e.via}</text>
            </g>
          );
        })}
        {members.map((m) => {
          const p = pos.get(m.id)!;
          const bad = m.status === "rejected" || m.status === "blocklisted";
          const on = m.id === selected;
          return (
            <g key={m.id} onClick={() => onSelect(m.id)} style={{ cursor: "pointer" }}>
              <circle cx={p.x} cy={p.y} r={on ? 13 : 10}
                      fill={bad ? "#a3231c" : "#2c6e5b"}
                      stroke={on ? "#16202c" : "none"} strokeWidth={2} />
              <text x={p.x} y={p.y - 20} textAnchor="middle"
                    fontFamily="Newsreader, serif" fontSize="13" fill="#16202c">{m.name}</text>
              <text x={p.x} y={p.y + 26} textAnchor="middle"
                    fontFamily="IBM Plex Mono, monospace" fontSize="9.5" fill="#8b97a3">
                {m.date} · {m.status}
              </text>
            </g>
          );
        })}
      </svg>
      <div style={{ maxWidth: "34ch" }}>
        <p className="note" style={{ marginTop: 0 }}>
          {members.length} applications, {flagged} of them rejected or
          blocklisted. Edges are labelled with the field that carries the link.
        </p>
        <p className="note">
          Where a chain runs A–B–C–D, the first and last records often share
          nothing at all. Pairwise matching sees unrelated strangers; the group
          only appears once the links are followed through.
        </p>
        <p className="note">Click a node to open that application.</p>
      </div>
    </div>
  );
}