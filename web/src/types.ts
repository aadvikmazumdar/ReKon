export interface Application {
  id: string; name: string; business: string;
  pan: string; gstin: string; phone: string;
  account: string; ifsc: string; mcc: string;
  address: string; city: string; pincode: string; email: string;
  date: string; amount: number; status: string;
  cluster: number; cohort: string;
  risk: number; isEvasion: boolean; level: number;
  priorRejections: number; clusterSize: number;
  breadth: number; bestLink: number; hasPrior: number;
  split: "train" | "test";
}

export interface EvidenceField { label: string; value: number; match: boolean }
export interface Contribution { feature: string; value: number; contribution: number }
export interface Detail {
  linkedTo: string; bits: number;
  fields: EvidenceField[]; contributions: Contribution[];
}
export interface ClusterEdge { a: string; b: string; score: number; via: string }
export interface Cluster { members: string[]; edges: ClusterEdge[] }

export interface Meta {
  generated: string; records: number; evasionRate: number; positives: number;
  edgeThreshold: number; blockThreshold: number; reviewThreshold: number;
  economicReviewThreshold: number;
  costFp: number; costFn: number; costReview: number;
  trainSize: number; testSize: number;
}

export interface Metrics {
  testAP: number; testAUC: number; linkageRecall: number;
  clusterPurity: number; contaminated: number; multiClusters: number;
  candidatePairs: number;
  evasionCurve: { level: number; rings: number; meanRecovered: number; fullyRecovered: number }[];
  thresholdSweep: { threshold: number; edges: number; precision: number; recall: number; purity: number; contaminated: number }[];
  costSensitivity: { ratio: number; threshold: number; blocked: number; recall: number; fpRate: number }[];
  queueBudget: { n: number; precision: number; recall: number }[];
  calibration: { predicted: number; observed: number; n: number }[];
  fsWeights: { field: string; m: number; u: number; agree: number; disagree: number }[];
  coefficients: { feature: string; coefficient: number }[];
}

export interface Dataset {
  meta: Meta; metrics: Metrics;
  applications: Application[];
  clusters: Record<string, Cluster>;
  details: Record<string, Detail>;
}

export type Action = "BLOCK" | "REVIEW" | "PASS";

export const decide = (risk: number, block: number, review: number): Action =>
  risk >= block ? "BLOCK" : risk >= review ? "REVIEW" : "PASS";

export const LEGITIMATE = new Set([
  "singleton", "grey", "hn_common_name", "hn_family_phone",
  "hn_honest_reapply", "hn_multi_founder", "hn_rebrand", "hn_shared_account",
]);

export const COHORT_LABEL: Record<string, string> = {
  singleton: "unrelated applicant",
  grey: "weak partial overlap",
  ring_origin: "first application of a ring",
  ring_evasion: "evasion attempt",
  hn_common_name: "same name, different person",
  hn_family_phone: "shared household phone",
  hn_honest_reapply: "honest re-application",
  hn_multi_founder: "one founder, two businesses",
  hn_rebrand: "business renamed",
  hn_shared_account: "shared settlement account",
};