import { useEffect, useMemo, useState } from "react";
import PlayerPanel from "./components/PlayerPanel";
import TableView from "./components/TableView";
import TimelineControls from "./components/TimelineControls";
import { ReplayData, Snapshot } from "./types";

const App = () => {
  const [replay, setReplay] = useState<ReplayData | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadReplay = async () => {
      try {
        setLoading(true);
        const response = await fetch("/replay.json", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(
            "Unable to load replay.json. Run demo_learned.py to generate it."
          );
        }
        const payload = (await response.json()) as ReplayData;
        if (!payload.snapshots.length) {
          throw new Error("Replay file does not contain any steps.");
        }
        setReplay(payload);
        setStepIndex(0);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    };

    loadReplay();
  }, []);

  const snapshot: Snapshot | null = useMemo(() => {
    if (!replay) {
      return null;
    }
    return replay.snapshots[Math.min(stepIndex, replay.snapshots.length - 1)];
  }, [replay, stepIndex]);

  if (loading) {
    return (
      <main className="app app--loading">
        <p>Loading replay...</p>
      </main>
    );
  }

  if (error || !replay || !snapshot) {
    return (
      <main className="app app--error">
        <p>{error ?? "Replay could not be loaded."}</p>
        <p className="hint">
          Tip: Run <code>python demo_learned.py --num-steps 40</code> to create a
          fresh replay file.
        </p>
      </main>
    );
  }

  const { metadata, snapshots } = replay;
  const playerPanels = snapshot.hands.map((hand, idx) => {
    const isModel = idx === 0;
    const actionCard = snapshot.actionCards[idx];
    const actionRow = snapshot.actionRows[idx];

    return (
      <PlayerPanel
        key={`player-${idx}`}
        title={isModel ? "Model" : `Player ${idx}`}
        hand={hand}
        winCount={snapshot.winCounts[idx]}
        totalPenalty={snapshot.totalPenalties[idx]}
        actionCard={actionCard}
        actionRow={actionRow}
        isModel={isModel}
        cardProbabilities={isModel ? snapshot.modelCardProbs : undefined}
        rowProbabilities={isModel ? snapshot.modelRowProbs : undefined}
      />
    );
  });

  return (
    <main className="app">
      <header className="app__header">
        <div>
          <h1>Take5 Replay Viewer</h1>
          <p>
            {metadata.numPlayers} players • {metadata.numSteps} simulated steps
            {metadata.seed !== null && ` • seed ${metadata.seed}`}
          </p>
        </div>
        <div className="app__status">
          <span>
            Step {snapshot.stepIndex + 1} / {snapshots.length}
          </span>
          <span>Game {snapshot.episodeIndex + 1}</span>
        </div>
      </header>

      <section className="players players--tabletop">{playerPanels}</section>

      <TableView snapshot={snapshot} />

      <TimelineControls
          currentStep={stepIndex}
          maxStep={snapshots.length - 1}
          onChange={setStepIndex}
        />
    </main>
  );
};

export default App;

