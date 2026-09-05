import { renderToString } from "react-dom/server";
import fs from "fs";
import path from "path";
import CaseDetail from "../src/components/CaseDetail";
import RingGraph from "../src/components/RingGraph";
import Metrics from "../src/components/Metrics";
import type { Dataset } from "../src/types";

const data: Dataset = JSON.parse(fs.readFileSync(path.resolve(process.cwd(), "public/data.json"), "utf8"));
const B = data.meta.blockThreshold, V = data.meta.reviewThreshold;

const pick = (f: (a: any) => boolean) => data.applications.find(f)!;
const cases = [
  ["top risk", data.applications[0]],
  ["honest reapply blocked", pick(a => a.cohort === "hn_honest_reapply" && a.risk >= B)],
  ["multi founder", pick(a => a.cohort === "hn_multi_founder" && a.hasPrior === 1)],
  ["singleton no link", pick(a => a.cohort === "singleton" && a.hasPrior === 0)],
  ["ring origin", pick(a => a.cohort === "ring_origin")],
  ["shared account", pick(a => a.cohort === "hn_shared_account")],
] as const;

let fails = 0;
for (const [label, app] of cases) {
  if (!app) { console.log(`SKIP  ${label}: no matching record`); continue; }
  try {
    const html = renderToString(<CaseDetail app={app} data={data} block={B} review={V} />);
    const ring = renderToString(
      <RingGraph data={data} clusterId={app.cluster} selected={app.id} onSelect={() => {}} />);
    console.log(`ok    ${label.padEnd(24)} ${app.id} risk=${app.risk.toFixed(3)} ` +
                `detail=${html.length}b ring=${ring.length}b`);
  } catch (e) {
    fails++; console.log(`FAIL  ${label}: ${(e as Error).message}`);
  }
}
try {
  const m = renderToString(<Metrics data={data} />);
  console.log(`ok    ${"metrics view".padEnd(24)} ${m.length}b`);
} catch (e) { fails++; console.log(`FAIL  metrics: ${(e as Error).message}`); }
console.log(fails === 0 ? "\nall components render" : `\n${fails} render failures`);