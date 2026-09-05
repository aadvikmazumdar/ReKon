import { useEffect, useMemo, useState } from "react";
import type { Application, Dataset } from "./types";
import { decide, LEGITIMATE } from "./types";
import CaseDetail from "./components/CaseDetail";
import RingGraph from "./components/RingGraph";
import Metrics from "./components/Metrics";

type View = "queue" | "metrics";
type Filter = "all" | "BLOCK" | "REVIEW" | "PASS";

export default function App() {
  const [data, setData] = useState<Dataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("queue");
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [block, setBlock] = useState<number | null>(null);
  const [review, setReview] = useState<number | null>(null);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`data.json responded ${r.status}`);
        return r.json() as Promise<Dataset>;
      })
      .then((d) => {
        setData(d);
        setBlock(d.meta.blockThreshold);
        setReview(d.meta.reviewThreshold);
        setSelected(d.applications[0]?.id ?? null);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const rows = useMemo(() => {
    if (!data || block === null || review === null) return [];
    const q = query.trim().toLowerCase();
    return data.applications.filter((a) => {
      if (filter !== "all" && decide(a.risk, block, review) !== filter) return false;
      if (!q) return true;
      return (
        a.name.toLowerCase().includes(q) ||
        a.id.toLowerCase().includes(q) ||
        a.business.toLowerCase().includes(q) ||
        a.pan.toLowerCase().includes(q)
      );
    });
  }, [data, filter, query, block, review]);

  const tally = useMemo(() => {
    if (!data || block === null || review === null) return null;
    let blocked = 0, reviewed = 0, falsePos = 0, caught = 0, evasions = 0;
    for (const a of data.applications) {
      const v = decide(a.risk, block, review);
      if (v === "BLOCK") blocked++;
      if (v === "REVIEW") reviewed++;
      if (a.isEvasion) {
        evasions++;
        if (v === "BLOCK") caught++;
      } else if (v === "BLOCK" && LEGITIMATE.has(a.cohort)) falsePos++;
    }
    return {
      blocked, reviewed, falsePos,
      recall: evasions ? (caught / evasions) * 100 : 0,
    };
  }, [data, block, review]);

  if (error) {
    return (
      <div className="loading">
        <p><b>Could not load data.json.</b> {error}</p>
        <p className="note">
          Generate it with <code>python3 -m rekon.export_web</code> from the
          project root; it is written to <code>web/public/data.json</code>.
        </p>
      </div>
    );
  }
  if (!data || block === null || review === null) {
    return <div className="loading">Loading case data…</div>;
  }

  const current = data.applications.find((a) => a.id === selected) ?? null;

  return (
    <div className="app">
      <header className="masthead">
        <div className="wordmark">ReKon<span> / underwriting risk</span></div>
        <div className="tagline">
          Is this applicant hiding a connection to a decision we already made?
        </div>
        <nav>
          <button data-on={view === "queue"} onClick={() => setView("queue")}>Cases</button>
          <button data-on={view === "metrics"} onClick={() => setView("metrics")}>Measurement</button>
        </nav>
      </header>

      {view === "queue" && (
        <div className="control">
          <label>
            Block above
            <input type="range" min={0.005} max={0.995} step={0.005} value={block}
                   onChange={(e) => {
                     const v = Number(e.target.value);
                     setBlock(v);
                     if (review > v) setReview(v);
                   }} />
            <span className="readout">{block.toFixed(3)}</span>
          </label>
          <label>
            Review above
            <input type="range" min={0.001} max={0.995} step={0.001} value={review}
                   onChange={(e) => setReview(Math.min(Number(e.target.value), block))} />
            <span className="readout">{review.toFixed(3)}</span>
          </label>
          <button className="reset" onClick={() => {
            setBlock(data.meta.blockThreshold);
            setReview(data.meta.reviewThreshold);
          }}>Reset to cost-optimal</button>
          {tally && (
            <div className="tally">
              <span className="v-BLOCK">blocked <b>{tally.blocked}</b></span>
              <span className="v-REVIEW">to review <b>{tally.reviewed}</b></span>
              <span>evasion caught <b>{tally.recall.toFixed(1)}%</b></span>
              <span className="v-BLOCK">legitimate blocked <b>{tally.falsePos}</b></span>
            </div>
          )}
        </div>
      )}

      {view === "metrics" ? (
        <Metrics data={data} />
      ) : (
        <div className="panes">
          <div className="queue">
            <div className="filters">
              {(["all", "BLOCK", "REVIEW", "PASS"] as Filter[]).map((f) => (
                <button key={f} data-on={filter === f} onClick={() => setFilter(f)}>
                  {f === "all" ? "All" : f[0] + f.slice(1).toLowerCase()}
                </button>
              ))}
            </div>
            <div className="search">
              <input value={query} placeholder="Search name, PAN or application id"
                     onChange={(e) => setQuery(e.target.value)} />
            </div>
            {rows.length === 0 && <div className="empty">No applications match.</div>}
            {rows.slice(0, 250).map((a: Application) => {
              const v = decide(a.risk, block, review);
              return (
                <button key={a.id} className="row" data-on={a.id === selected}
                        onClick={() => setSelected(a.id)}>
                  <div className="who">{a.name}</div>
                  <div className={`score v-${v}`}>{a.risk.toFixed(2)}</div>
                  <div className="meta">{a.id} · {a.date} · {a.status}</div>
                </button>
              );
            })}
            {rows.length > 250 && (
              <div className="empty">{rows.length - 250} more not shown.</div>
            )}
          </div>

          {current ? (
            <div style={{ flex: 1, overflowY: "auto" }}>
              <CaseDetail app={current} data={data} block={block} review={review} />
              <div style={{ padding: "0 2rem 3rem" }}>
                <h3 className="section">The group, as a graph</h3>
                <RingGraph data={data} clusterId={current.cluster}
                           selected={current.id} onSelect={setSelected} />
              </div>
            </div>
          ) : (
            <div className="detail"><p className="note">Select an application.</p></div>
          )}
        </div>
      )}
    </div>
  );
}