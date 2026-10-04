import { useState } from "react";

import Header from "./components/Header";
import LanguageSelector from "./components/LanguageSelector";
import PatientInput from "./components/PatientInput";
import ClinicalSummary from "./components/ClinicalSummary";

import type { ClinicalIntake } from "./types/clinical";


function App() {
  const [language, setLanguage] = useState("en");

  const [patientText, setPatientText] = useState("");

  const [result, setResult] =
    useState<ClinicalIntake | null>(null);

  const [loading, setLoading] = useState(false);

const handleAnalyze = async () => {
  if (!patientText.trim()) {
    return;
  }

  setLoading(true);
  setResult(null);

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/api/intake",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: patientText,
          language: language,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(
        `Request failed: ${response.status}`
      );
    }

    const data: ClinicalIntake =
      await response.json();

    setResult(data);
  } catch (error) {
    console.error(
      "Failed to structure intake:",
      error
    );
  } finally {
    setLoading(false);
  }
};

  const handleLanguageChange = (
    newLanguage: string
  ) => {
    setLanguage(newLanguage);

    // Prevent a result from a previous language
    // remaining visible.
    setResult(null);
  };

  return (
    <div className="app">
      <Header />

      <main className="main">
        <section className="intro">
          <span className="eyebrow">
            Offline-first clinical support
          </span>

          <h2>
            Turn patient words into structured
            clinical information.
          </h2>

          <p>
            Capture symptoms, durations,
            medications and allergies while
            preserving uncertainty and missing
            information.
          </p>
        </section>

        <div className="workspace">
          <section className="intakePanel">
            <div className="panelHeader">
              <span className="step">
                01
              </span>

              <div>
                <h2>Patient Intake</h2>

                <p>
                  Record the patient's statement
                  in their preferred language.
                </p>
              </div>
            </div>

            <LanguageSelector
              value={language}
              onChange={handleLanguageChange}
            />

            <PatientInput
              value={patientText}
              onChange={setPatientText}
              onAnalyze={handleAnalyze}
              loading={loading}
            />
          </section>

          {result ? (
            <ClinicalSummary
              data={result}
            />
          ) : (
            <section className="emptyState">
              <div className="emptyIcon">
                ✦
              </div>

              <h3>
                Clinical summary
              </h3>

              <p>
                The structured intake will appear
                here after processing the patient
                statement.
              </p>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;