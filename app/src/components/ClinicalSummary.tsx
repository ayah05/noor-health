import {
  AlertCircle,
  CheckCircle2,
  CircleHelp,
  Clock3,
  Pill,
  ShieldCheck,
} from "lucide-react";

import type {
  ClinicalIntake,
  Symptom,
} from "../types/clinical";

interface Props {
  data: ClinicalIntake;
}

function formatDuration(symptom: Symptom) {
  if (!symptom.duration) {
    return "Not reported";
  }

  const { value, unit } = symptom.duration;

  return `${value} ${unit}`;
}

function statusLabel(status: Symptom["status"]) {
  switch (status) {
    case "present":
      return "Present";

    case "absent":
      return "Absent";

    case "uncertain":
      return "Uncertain";
  }
}

function informationStatusLabel(
  status: "unknown" | "none" | "reported"
) {
  switch (status) {
    case "unknown":
      return "Not reported";

    case "none":
      return "None reported";

    case "reported":
      return "Reported";
  }
}

function formatMissingInformation(
  item: string
) {
  switch (item) {
    case "duration":
      return "symptom duration";

    case "medications":
      return "current medications";

    case "allergies":
      return "allergies";

    default:
      return item.replaceAll("_", " ");
  }
}

export default function ClinicalSummary({
  data,
}: Props) {
  return (
    <section className="summary">
      <div className="summaryHeader">
        <div>
          <span className="eyebrow">
            Structured intake
          </span>

          <h2>Clinical Summary</h2>
        </div>

        <div className="reviewBadge">
          Human review required
        </div>
      </div>

      <div className="summaryCard chiefComplaint">
        <span className="cardLabel">
          Chief complaint
        </span>

        <strong>
          {data.chief_complaint}
        </strong>
      </div>

      <div className="summaryCard">
        <h3>Symptoms</h3>

        <div className="symptomList">
          {data.symptoms.map(
            (symptom, index) => (
              <div
                className="symptomRow"
                key={`${symptom.name}-${index}`}
              >
                <div className="symptomName">
                  {symptom.status ===
                    "present" && (
                    <CheckCircle2 size={18} />
                  )}

                  {symptom.status ===
                    "absent" && (
                    <AlertCircle size={18} />
                  )}

                  {symptom.status ===
                    "uncertain" && (
                    <CircleHelp size={18} />
                  )}

                  <span>
                    {symptom.name}
                  </span>
                </div>

                <span
                  className={`status ${symptom.status}`}
                >
                  {statusLabel(
                    symptom.status
                  )}
                </span>

                <span className="duration">
                  <Clock3 size={15} />

                  {formatDuration(symptom)}
                </span>
              </div>
            )
          )}
        </div>
      </div>

      <div className="informationGrid">
        <div className="summaryCard">
          <div className="cardTitle">
            <Pill size={19} />

            <h3>Medications</h3>
          </div>

          <p>
            <strong>
              {informationStatusLabel(
                data.medications.status
              )}
            </strong>
          </p>

          {data.medications.status ===
            "reported" &&
            data.medications.items.length >
              0 && (
              <p className="informationItems">
                {data.medications.items.join(
                  ", "
                )}
              </p>
            )}
        </div>

        <div className="summaryCard">
          <div className="cardTitle">
            <ShieldCheck size={19} />

            <h3>Allergies</h3>
          </div>

          <p>
            <strong>
              {informationStatusLabel(
                data.allergies.status
              )}
            </strong>
          </p>

          {data.allergies.status ===
            "reported" &&
            data.allergies.items.length >
              0 && (
              <p className="informationItems">
                {data.allergies.items.join(
                  ", "
                )}
              </p>
            )}
        </div>
      </div>

      {data.missing_information.length >
        0 && (
        <div className="missingCard">
          <AlertCircle size={21} />

          <div>
            <strong>
              More information needed
            </strong>

            <p>
              Follow up about{" "}
              {data.missing_information
                .map(
                  formatMissingInformation
                )
                .join(", ")}
              .
            </p>
          </div>
        </div>
      )}

      {data.missing_information.length ===
        0 && (
        <div className="completeCard">
          <CheckCircle2 size={21} />

          <div>
            <strong>
              Intake information complete
            </strong>

            <p>
              No required intake fields are
              currently missing.
            </p>
          </div>
        </div>
      )}

      <p className="disclaimer">
        Noor Health structures
        patient-reported information. It does
        not provide a diagnosis. Clinical
        review is required.
      </p>
    </section>
  );
}