import CardStrip from "./CardStrip";
import { Snapshot } from "../types";

interface TableViewProps {
  snapshot: Snapshot;
}

const TableView = ({ snapshot }: TableViewProps) => {
  const rows = snapshot.table;

  return (
    <section className="table-view">
      <header>
        <h2>Table Center</h2>
        <p>
          Total penalties this game:{" "}
            {snapshot.totalPenalties.reduce((sum, value) => sum + value, 0).toFixed(0)}
        </p>
      </header>
      <div className="table-view__rows">
        {rows.map((row, idx) => (
          <div key={`row-${idx}`} className="table-view__row">
            <div className="table-view__row-header">
              <span>Row {idx + 1}</span>
              <span>Penalty {snapshot.rowPenalties[idx]}</span>
            </div>
            <CardStrip cards={row.filter((card) => card !== 0)} />
          </div>
        ))}
      </div>
    </section>
  );
};

export default TableView;

