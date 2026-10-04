import {
  Mic,
  Sparkles,
} from "lucide-react";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  loading: boolean;
}

export default function PatientInput({
  value,
  onChange,
  onAnalyze,
  loading,
}: Props) {
  return (
    <div className="patientInput">
      <div className="field">
        <label htmlFor="patientStatement">
          Patient statement
        </label>

        <textarea
          id="patientStatement"
          value={value}
          onChange={(event) =>
            onChange(event.target.value)
          }
          placeholder="Enter the patient's own words..."
          rows={7}
          disabled={loading}
        />
      </div>

      <div className="inputActions">
        <button
          className="secondaryButton"
          type="button"
          disabled
          title="Voice input coming next"
        >
          <Mic size={18} />
          Voice
        </button>

        <button
          className="primaryButton"
          type="button"
          onClick={onAnalyze}
          disabled={!value.trim() || loading}
        >
          <Sparkles size={18} />

          {loading
            ? "Processing..."
            : "Structure intake"}
        </button>
      </div>
    </div>
  );
}