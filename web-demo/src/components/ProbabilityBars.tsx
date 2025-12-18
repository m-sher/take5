interface ProbabilityEntry {
  label: string;
  probability: number;
}

interface ProbabilityBarsProps {
  entries: ProbabilityEntry[];
}

const ProbabilityBars = ({ entries }: ProbabilityBarsProps) => {
  if (!entries.length) {
    return null;
  }

  return (
    <div className="probability-bars">
      {entries.map((entry) => {
        const heightPercent = Math.min(100, Math.max(4, entry.probability * 100));

        return (
          <div key={entry.label} className="probability-bars__column">
            <span className="probability-bars__value">
              {entry.probability.toFixed(2)}
            </span>
            <div
              className="probability-bars__bar"
              data-card={entry.label}
              aria-label={`Probability ${entry.probability.toFixed(2)}`}
              style={{ height: `${heightPercent}%` }}
            />
          </div>
        );
      })}
    </div>
  );
};

export default ProbabilityBars;

