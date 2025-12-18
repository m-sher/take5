export interface Snapshot {
  stepIndex: number;
  episodeIndex: number;
  hands: number[][];
  table: number[][];
  rowPenalties: number[];
  totalPenalties: number[];
  winCounts: number[];
  actionCards: number[];
  actionRows: number[];
  modelCardProbs: number[];
  modelRowProbs: number[];
}

export interface ReplayMetadata {
  numPlayers: number;
  numSteps: number;
  seed: number | null;
}

export interface ReplayData {
  metadata: ReplayMetadata;
  snapshots: Snapshot[];
}

