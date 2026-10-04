export type SymptomStatus =
  | "present"
  | "absent"
  | "uncertain";

export type InformationStatus =
  | "unknown"
  | "none"
  | "reported";

export type DurationUnit =
  | "hours"
  | "days"
  | "weeks";

export interface SymptomDuration {
  value: number;
  unit: DurationUnit;
}

export interface Symptom {
  name: string;
  status: SymptomStatus;
  duration: SymptomDuration | null;
}

export interface InformationGroup {
  status: InformationStatus;
  items: string[];
}

export interface ClinicalIntake {
  chief_complaint: string;

  symptoms: Symptom[];

  medications: InformationGroup;

  allergies: InformationGroup;

  missing_information: string[];
}