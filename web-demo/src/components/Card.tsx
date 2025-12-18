interface CardProps {
  value: number;
}

const Card = ({ value }: CardProps) => {
  return (
    <div className="card">
      <span>{value}</span>
    </div>
  );
};

export default Card;

