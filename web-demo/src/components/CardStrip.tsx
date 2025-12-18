import Card from "./Card";

interface CardStripProps {
  cards: number[];
  align?: "left" | "center";
}

const CardStrip = ({ cards, align = "left" }: CardStripProps) => {
  if (!cards.length) {
    return <div className="empty-strip">No cards</div>;
  }

  return (
    <div
      className="card-strip"
      style={{ justifyContent: align === "center" ? "center" : "flex-start" }}
    >
      {cards.map((card, idx) => (
        <Card key={`${card}-${idx}`} value={card} />
      ))}
    </div>
  );
};

export default CardStrip;

