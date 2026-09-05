import type { Application, Dataset, Action } from "../types";
import { decide, COHORT_LABEL } from "../types";

interface Props {
  app: Application;
  data: Dataset;
  block: number;
  review: number;
}

const rupees = (n: number) => "\u20B9" + n.toLocaleString("en-IN");

export default function CaseDetail({ app, data, block, review }: Props) {
  const verdict: Action = decide(app.risk, block, review);
  const detail = data.details[app.id];
  const cluster = data.clusters[String(app.cluster)];
  const members = (cluster?.members ?? [app.id])
    .map((id) => data.applications.find((a) => a.id === id))
    .filter((a): a is Application => Boolean(a))
    .sort((a, b) => a.date.localeCompare(b.date));

  const priorBad = members.filter(
    (m) => m.date < app.date && (m.status === "rejected" || m.status === "blocklisted"),
  ).length;

  const agree = detail?.fields.filter((x) => x.match) ?? [];
  const differ = detail?.fields.filter((x) => !x.match) ?? [];
  const maxAbs = Math.max(1, ...(detail?.contributions ?? []).map((c) => Math.abs(c.contribution)));

  // A legitimate applicant that gets blocked is a false positive, and the page
  // should say so rather than inventing a justification for the decision.
  const legitimate = app.cohort === "hn_honest_reapply" && !app.isEvasion;
  const falsePositive = legitimate && verdict === "BLOCK";

  return (
    <div className="detail">
      <div className="case-head">
        <div>
          <h2>{app.name}</h2>
          <div style={{ color: "var(--muted)" }}>
            {app.business} · {COHORT_LABEL[app.cohort] ?? app.cohort}
          </div>
        </div>
        <div className="verdict">
          <div className={`n v-${verdict}`}>{app.risk.toFixed(3)}</div>
          <div className={`chip v-${verdict}`}>{verdict}</div>
        </div>
      </div>

      <div className="identity">
        <div><b>{app.id}</b> · applied {app.date}</div>
        <div>PAN <b>{app.pan}</b></div>
        <div>GSTIN <b>{app.gstin}</b></div>
        <div>phone <b>{app.phone}</b></div>
        <div>account <b>{app.account}</b> · {app.ifsc}</div>
        <div>{app.city} {app.pincode} · {rupees(app.amount)}</div>
      </div>

      <h3 className="section">
        Linked group<small>{members.length} application{members.length === 1 ? "" : "s"}</small>
      </h3>
      <table className="timeline">
        <tbody>
          {members.map((m) => (
            <tr key={m.id} data-self={m.id === app.id}>
              <td className="d">{m.date}</td>
              <td className="i">{m.id}</td>
              <td>{m.name}</td>
              <td className="st">{m.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        {priorBad === 0
          ? "No earlier application in this group was rejected or blocklisted."
          : `${priorBad} earlier application${priorBad === 1 ? " was" : "s were"} rejected or blocklisted.`}
      </p>

      {detail && (
        <>
          <h3 className="section">
            Strongest earlier link
            <small>{detail.linkedTo} · {detail.bits.toFixed(1)} bits</small>
          </h3>
          <div className="ledger">
            <div className="agree">
              <h4>What matches</h4>
              <ul>
                {agree.map((x) => (
                  <li key={x.label}><span>{x.label}</span><span>{x.value.toFixed(2)}</span></li>
                ))}
                {agree.length === 0 && <li><span style={{ color: "var(--faint)" }}>nothing</span><span /></li>}
              </ul>
            </div>
            <div className="differ">
              <h4>What differs</h4>
              <ul>
                {differ.map((x) => (
                  <li key={x.label}><span>{x.label}</span><span>{x.value.toFixed(2)}</span></li>
                ))}
                {differ.length === 0 && <li><span style={{ color: "var(--faint)" }}>nothing</span><span /></li>}
              </ul>
            </div>
          </div>

          <h3 className="section">What drove the score</h3>
          <div className="contribs">
            {detail.contributions.map((c) => {
              const w = (Math.abs(c.contribution) / maxAbs) * 50;
              return (
                <div className="bar" key={c.feature}>
                  <div className="feat">{c.feature}</div>
                  <div className="track">
                    <div className="mid" />
                    <div
                      className={`fill ${c.contribution >= 0 ? "pos" : "neg"}`}
                      style={{ width: `${w}%` }}
                    />
                  </div>
                  <div className="num">{c.contribution >= 0 ? "+" : ""}{c.contribution.toFixed(2)}</div>
                </div>
              );
            })}
          </div>
          <p className="note">
            Bars to the right raise the score, to the left lower it. These are the
            model&apos;s own coefficients, not an approximation of them.
          </p>
        </>
      )}

      <div className={`verdict-note${falsePositive ? " fp" : ""}`}>
        {falsePositive ? (
          <>
            <b>Known false positive.</b> {app.breadth} field
            {app.breadth === 1 ? "" : "s"} differ from the earlier application,
            which is within the range an ordinary applicant produces months
            later. The evidence cannot separate a light disguise from a changed
            phone number, so ReKon blocks a legitimate merchant. This band is
            measured, not hidden.
          </>
        ) : priorBad === 0 ? (
          <>
            <b>No prior risk decision in this group.</b> Linkage on its own is not
            risk: the same person may legitimately apply more than once.
          </>
        ) : app.breadth === 0 ? (
          <>
            <b>Linked, but nothing was altered.</b> Every compared field matches
            the earlier application, which fits an open re-application rather
            than an attempt to look like someone new.
          </>
        ) : (
          <>
            <b>Linked to {priorBad} prior rejection{priorBad === 1 ? "" : "s"} with{" "}
            {app.breadth} altered field{app.breadth === 1 ? "" : "s"}.</b>{" "}
            Concealment is the signal here, not the linkage itself.
          </>
        )}
      </div>
    </div>
  );
}