interface TimelineControlsProps {
  currentStep: number;
  maxStep: number;
  onChange: (value: number) => void;
}

const TimelineControls = ({
  currentStep,
  maxStep,
  onChange
}: TimelineControlsProps) => {
  return (
    <div className="timeline">
      <button
        type="button"
        onClick={() => onChange(Math.max(0, currentStep - 1))}
        disabled={currentStep === 0}
      >
        Prev
      </button>
      <input
        type="range"
        min={0}
        max={maxStep}
        value={currentStep}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <button
        type="button"
        onClick={() => onChange(Math.min(maxStep, currentStep + 1))}
        disabled={currentStep === maxStep}
      >
        Next
      </button>
    </div>
  );
};

export default TimelineControls;

