// Shared types for the System One playground.

export type S1QuestionType = 'choice' | 'noul' | 'score';

export interface ChoiceCriterion {
  key: string;
  label: string;
}

export interface S1Question {
  id: string;
  type: S1QuestionType;
  instructions: string;
  /** choice: option key -> description */
  criteria: ChoiceCriterion[];
  /** score: ordered level descriptions (2..10) */
  levels: string[];
}

/** One parsed answer from the Jev-native answers map. */
export interface S1Answer {
  id: string;
  type: 'choice' | 'noul' | 'score';
  choice?: string;
  score?: number;
  /** noul: probability of yes (no confidence by contract) */
  noul?: number;
  probabilities?: Record<string, number>;
  confidence?: number;
  legend?: Record<string, string>;
  abstained?: boolean;
  degraded?: boolean;
}

export interface S1LegacyResult {
  model: string;
  value: string;
  confidence: number;
  abstained: boolean;
  prompt_tokens: number;
  degraded?: boolean;
  engine?: string;
}

export interface S1RunResponse {
  model: string;
  answers: Record<string, any>;
  usage: { input_tokens: number; output_tokens: number };
  // single-question legacy top-level fields
  value?: string;
  confidence?: number;
  engine?: string;
  prompt_tokens?: number;
  degraded?: boolean;
  abstained?: boolean;
  // multi-question legacy results
  results?: S1LegacyResult[];
}

export interface TimingInfo {
  queueMs: number | null;
  decisionMs: number | null;
  wallMs: number;
}

export interface S1Run {
  id: string;
  ts: number;
  modelId: string;
  label: string;
  totalMs: number;
  timing: TimingInfo;
  engine: string | null;
  inputTokens: number | null;
  answers: S1Answer[];
  /** request snapshot for restore */
  request: S1RequestState;
}

export interface S1RequestState {
  stateText: string;
  stateIsJson: boolean;
  questions: S1Question[];
  minConfidence: number;
  readoutPath: string;
}

export interface LoadedModelInfo {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  architecture?: string;
  gpu: boolean;
  path?: string;
}

export interface VRAMStatus {
  total_mb: number;
  free_mb: number;
  used_mb: number;
}
