interface RowProbabilityStripProps {
  entries: { label: string; probability: number }[];
}

const RowProbabilityStrip = ({ entries }: RowProbabilityStripProps) => {
  if (!entries.length) {
    return null;
  }

  const maxProb = Math.max(...entries.map((entry) => entry.probability), 0.001);

  return (
    <div className="row-strip">
      {entries.map((entry) => {
        const width = (entry.probability / maxProb) * 100;
        return (
          <div key={entry.label} className="row-strip__entry">
            <span>Row {entry.label}</span>
            <div className="row-strip__bar">
              <div style={{ width: `${width}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default RowProbabilityStrip;

