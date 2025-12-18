import CardStrip from "./CardStrip";
import ProbabilityBars from "./ProbabilityBars";
import RowProbabilityStrip from "./RowProbabilityStrip";

interface PlayerPanelProps {
  title: string;
  hand: number[];
  winCount: number;
  totalPenalty: number;
  actionCard: number;
  actionRow: number;
  isModel?: boolean;
  cardProbabilities?: number[];
  rowProbabilities?: number[];
}

const PlayerPanel = ({
  title,
  hand,
  winCount,
  totalPenalty,
  actionCard,
  actionRow,
  isModel = false,
  cardProbabilities,
  rowProbabilities
}: PlayerPanelProps) => {
  const visibleCards = hand
    .map((value, index) => ({ value, index }))
    .filter(({ value }) => value !== 0)
    .sort((a, b) => a.value - b.value);

  const cardProbEntries =
    isModel && cardProbabilities
      ? visibleCards.map(({ value, index }) => ({
          label: String(value),
          probability: cardProbabilities[index] ?? 0
        }))
      : [];

  const rowProbEntries =
    isModel && rowProbabilities
      ? rowProbabilities.map((probability, idx) => ({
          label: String(idx + 1),
          probability
        }))
      : [];

  return (
    <section className={`panel ${isModel ? "panel--model" : ""}`}>
      <header>
        <div>
          <h2>{title}</h2>
          <p>
            Wins: {winCount} • Penalty: {totalPenalty.toFixed(0)}
          </p>
        </div>
        <span className="panel__action">
          {actionCard > 0
            ? `Playing ${actionCard} & Row ${actionRow + 1}`
            : "No playable card"}
        </span>
      </header>

      {isModel && cardProbEntries.length > 0 && (
        <ProbabilityBars entries={cardProbEntries} />
      )}

      <CardStrip
        cards={visibleCards.map(({ value }) => value)}
        align={isModel ? "center" : "left"}
      />

      {isModel && rowProbEntries.length > 0 && (
        <RowProbabilityStrip entries={rowProbEntries} />
      )}
    </section>
  );
};

export default PlayerPanel;

