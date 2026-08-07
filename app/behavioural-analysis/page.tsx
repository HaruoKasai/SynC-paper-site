import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Behavioural analysis code | SynC",
  description:
    "Python code and demonstration data for behavioural analyses used in the SynC manuscript.",
};

const analyses = [
  {
    id: "food-approach",
    label: "Food-approach response",
    title: "Distance and velocity around food placement",
    description:
      "Calculates the distance from the tracked head centre to the food location and the velocity of the body centre. Results are grouped by time relative to A/C administration and summarised across mice.",
    details: [
      "Analysis window: 30 s before to 90 s after food placement",
      "AUC window: 30–90 s after food placement",
      "Manuscript scale: 0.76 mm per pixel",
      "Outputs: analysis PDF and source-data CSV files for curves and AUC values",
    ],
    script: "edf8_food.py",
  },
  {
    id: "laser-response",
    label: "Laser response in the open field",
    title: "Movement relative to laser stimulation",
    description:
      "Uses DeepLabCut tracking and synchronisation files to measure distance from the laser zone, velocity, and orientation-related responses around each laser event. The manuscript settings use a per-mouse average as the statistical basis.",
    details: [
      "Analysis window: 5 s before to 10 s after laser onset",
      "Laser-zone radius: 75 pixels; orientation threshold: 30°",
      "AUC window: 5–10 s after laser onset",
      "Outputs: overview PDF and source-data CSV files for curves and AUC values",
    ],
    script: "edf8_laser.py",
  },
];

export default function BehaviouralAnalysisPage() {
  return (
    <main>
      <nav className="nav" aria-label="Page navigation">
        <a className="brand" href="/">
          SynC
        </a>
        <div className="navLinks">
          <a href="/">Home</a>
          <a href="/python-code">Statistical code</a>
          <a
            href="https://github.com/HaruoKasai/SynC-paper-site"
            target="_blank"
            rel="noreferrer"
          >
            GitHub repository ↗
          </a>
        </div>
      </nav>

      <header className="subpageHero">
        <p className="eyebrow">Behavioural analysis</p>
        <h1>Food-approach and laser-response workflows</h1>
        <p className="lead">
          Python scripts used to analyse open-field behaviour from processed
          DeepLabCut tracking data, together with a demonstration dataset.
        </p>
      </header>

      <section className="scriptCatalogue" aria-label="Behavioural scripts">
        {analyses.map((analysis) => (
          <article className="scriptCard" id={analysis.id} key={analysis.id}>
            <div>
              <p className="sectionLabel">{analysis.label}</p>
              <h2>{analysis.title}</h2>
              <p>{analysis.description}</p>
              <ul className="analysisDetails">
                {analysis.details.map((detail) => (
                  <li key={detail}>{detail}</li>
                ))}
              </ul>
            </div>
            <div className="filePanel">
              <span>Analysis file</span>
              <strong>{analysis.script}</strong>
              <div className="fileLinks">
                <a href={`/code/${analysis.script}`} download>
                  Download Python
                </a>
                <a
                  href="https://github.com/HaruoKasai/SynC-paper-site/tree/main/public/data/edf8_demo"
                  target="_blank"
                  rel="noreferrer"
                >
                  Browse demonstration data ↗
                </a>
              </div>
            </div>
          </article>
        ))}
      </section>

      <section className="repositoryCallout">
        <div>
          <p className="sectionLabel">Demonstration data</p>
          <h2>Control and SynC example datasets</h2>
          <p className="calloutCopy">
            The demonstration folder contains processed tracking tables,
            event timing, food locations, laser time points, synchronisation
            files, and analysis configuration for the included example mice.
          </p>
        </div>
        <a
          className="repoLink"
          href="https://github.com/HaruoKasai/SynC-paper-site/tree/main/public/data/edf8_demo"
          target="_blank"
          rel="noreferrer"
        >
          Open data folder on GitHub ↗
        </a>
      </section>

      <footer>
        <span>SynC behavioural analysis code</span>
        <a href="/">Back to analysis resources</a>
      </footer>
    </main>
  );
}
