import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Dataset } from "../types";

const AXIS = { stroke: "#8b97a3", fontSize: 12, fontFamily: "IBM Plex Mono, monospace" };

export default function Metrics({ data }: { data: Dataset }) {
  const m = data.metrics;
  const meta = data.meta;

  return (
    <div className="metrics">
      <div className="headline">
        <div>
          <div className="n">{(m.linkageRecall * 100).toFixed(1)}%</div>
          <div className="l">of true same-operator pairs recovered, at 100% precision</div>
        </div>
        <div>
          <div className="n">{m.testAP.toFixed(3)}</div>
          <div className="l">average precision on the held-out split</div>
        </div>
        <div>
          <div className="n">{m.clusterPurity.toFixed(3)}</div>
          <div className="l">cluster purity ({m.contaminated} contaminated of {m.multiClusters})</div>
        </div>
        <div>
          <div className="n">{meta.evasionRate.toFixed(2)}%</div>
          <div className="l">evasion rate across {meta.records.toLocaleString()} applications</div>
        </div>
      </div>

      <h3 className="section">How much evasion effort it takes to break us</h3>
      <p className="note">
        Every figure here comes from data we generated, so the obvious objection is
        that we built the test and then passed it. Because the generator is ours,
        evasion effort can be dialled from one to five and the failure measured.
        The claim is not a score; it is a breaking point.
      </p>
      <div style={{ height: 280, marginTop: "1rem" }}>
        <ResponsiveContainer>
          <LineChart data={m.evasionCurve} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
            <CartesianGrid stroke="#e6eaee" />
            <XAxis dataKey="level" tick={AXIS}
                   label={{ value: "evasion level", position: "insideBottom", offset: -12, fill: "#5c6b7a", fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={AXIS} unit="%" />
            <Tooltip contentStyle={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 12 }} />
            <Line type="monotone" dataKey="meanRecovered" stroke="#2b4c7e" strokeWidth={2}
                  dot={{ r: 4 }} name="ring recovered" />
            <Line type="monotone" dataKey="fullyRecovered" stroke="#a3231c" strokeWidth={2}
                  strokeDasharray="4 3" dot={{ r: 4 }} name="fully recovered" />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <table className="data" style={{ marginTop: "1rem" }}>
        <thead>
          <tr><th>Level</th><th>Rings</th><th>Mean recovered</th><th>Fully recovered</th></tr>
        </thead>
        <tbody>
          {m.evasionCurve.map((r) => (
            <tr key={r.level}>
              <td>{r.level}</td><td>{r.rings}</td>
              <td>{r.meanRecovered.toFixed(1)}%</td><td>{r.fullyRecovered.toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="grid2" style={{ marginTop: "2.5rem" }}>
        <div>
          <h3 className="section">Link threshold and contamination</h3>
          <p className="note">
            One spurious edge welds two unrelated groups together, so a loose
            threshold does not merely add noise — it destroys the groups.
          </p>
          <table className="data" style={{ marginTop: ".8rem" }}>
            <thead>
              <tr><th>Threshold</th><th>Edges</th><th>Precision</th><th>Purity</th><th>Bad clusters</th></tr>
            </thead>
            <tbody>
              {m.thresholdSweep.map((r) => (
                <tr key={r.threshold}
                    style={r.threshold === meta.edgeThreshold ? { background: "#eef2f7" } : undefined}>
                  <td>{r.threshold.toFixed(2)}</td><td>{r.edges}</td>
                  <td>{r.precision.toFixed(3)}</td><td>{r.purity.toFixed(3)}</td>
                  <td>{r.contaminated}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h3 className="section">Calibration</h3>
          <p className="note">
            A score of 0.25 should mean a quarter of such applications really are
            evasion. Points on the diagonal are honest probabilities.
          </p>
          <div style={{ height: 240, marginTop: ".8rem" }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid stroke="#e6eaee" />
                <XAxis type="number" dataKey="predicted" domain={[0, "dataMax"]} tick={AXIS}
                       label={{ value: "predicted", position: "insideBottom", offset: -12, fill: "#5c6b7a", fontSize: 12 }} />
                <YAxis type="number" dataKey="observed" domain={[0, "dataMax"]} tick={AXIS} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#c3cbd4" strokeDasharray="4 3" />
                <Tooltip contentStyle={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 12 }} />
                <Scatter data={m.calibration} fill="#2b4c7e" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="section">Cost decides the threshold</h3>
          <p className="note">
            Blocking is worth it when p·cost(missed evader) exceeds
            (1−p)·cost(blocked merchant). The operating point follows from the
            ratio rather than from preference.
          </p>
          <table className="data" style={{ marginTop: ".8rem" }}>
            <thead>
              <tr><th>Miss : block</th><th>Threshold</th><th>Recall</th><th>False positives</th></tr>
            </thead>
            <tbody>
              {m.costSensitivity.map((r) => (
                <tr key={r.ratio}>
                  <td>{r.ratio}×</td><td>{r.threshold.toFixed(4)}</td>
                  <td>{r.recall.toFixed(1)}%</td><td>{r.fpRate.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h3 className="section">A day&apos;s review queue</h3>
          <p className="note">
            Underwriter capacity, not model accuracy, is what limits a risk
            system in practice.
          </p>
          <table className="data" style={{ marginTop: ".8rem" }}>
            <thead><tr><th>Queue depth</th><th>Precision</th><th>Recall</th></tr></thead>
            <tbody>
              {m.queueBudget.map((r) => (
                <tr key={r.n}>
                  <td>top {r.n}</td><td>{r.precision.toFixed(1)}%</td><td>{r.recall.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h3 className="section">Evidence weights, estimated not chosen</h3>
          <p className="note">
            Fitted by expectation-maximisation without labels. A field that
            agrees often by chance earns little, so name frequency is handled
            without a rule for it.
          </p>
          <table className="data" style={{ marginTop: ".8rem" }}>
            <thead><tr><th>Field</th><th>m</th><th>u</th><th>Bits when it agrees</th></tr></thead>
            <tbody>
              {m.fsWeights.slice(0, 10).map((r) => (
                <tr key={r.field}>
                  <td>{r.field}</td><td>{r.m.toFixed(3)}</td>
                  <td>{r.u.toFixed(4)}</td><td>{r.agree.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h3 className="section">Model coefficients</h3>
          <p className="note">
            Negative weights are exculpatory: a matching PAN or settlement
            account argues that nothing was hidden. Their signs are asserted on
            every build.
          </p>
          <table className="data" style={{ marginTop: ".8rem" }}>
            <thead><tr><th>Feature</th><th>Weight</th></tr></thead>
            <tbody>
              {m.coefficients.map((r) => (
                <tr key={r.feature}>
                  <td>{r.feature}</td>
                  <td style={{ color: r.coefficient >= 0 ? "var(--block)" : "var(--pass)" }}>
                    {r.coefficient >= 0 ? "+" : ""}{r.coefficient.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}