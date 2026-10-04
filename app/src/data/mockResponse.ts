import type { ClinicalIntake } from "../types/clinical";

export const mockClinicalResponse: ClinicalIntake = {
  chief_complaint: "cough",

  symptoms: [
    {
      name: "cough",
      status: "present",
      duration: {
        value: 4,
        unit: "days",
      },
    },
    {
      name: "fever",
      status: "absent",
      duration: null,
    },
    {
      name: "shortness of breath",
      status: "uncertain",
      duration: null,
    },
  ],

  medications: {
    status: "reported",
    items: ["ibuprofen"],
  },

  allergies: {
    status: "unknown",
    items: [],
  },

  missing_information: [
    "allergies",
  ],
};