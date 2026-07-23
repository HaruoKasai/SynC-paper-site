import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SynC | Analysis code",
  description:
    "Analysis code and statistical methods supporting the SynC manuscript.",
};

const codeItems = [
  {
    title: "EEG analysis",
    description:
      "Scripts used for processing and analysing electrophysiological recordings.",
    status: "Available",
    href: "https://github.com/tkssawada/SynC",
  },
  {
    title: "Imaging analysis",
    description:
      "Code used for calcium-imaging and spine-analysis workflows will be organised here.",
    status: "In preparation",
  },
  {
    title: "Behavioural analysis",
    description:
      "Scripts used to quantify behavioural state and task performance will be added here.",
    status: "In preparation",
  },
];

export default function Home() {
  return (
    <main>
      <nav className="nav" aria-label="Primary navigation">
        <a className="brand" href="#top">
          SynC
        </a>
        <div className="navLinks">
          <a href="#code">Code</a>
          <a href="#statistics">Statistics</a>
          <a href="#citation">Citation</a>
        </div>
      </nav>

      <section className="hero" id="top">
        <p className="eyebrow">Code and analysis resources</p>
        <h1>
          Rapid associative spine enlargement is required for cognitive
          function and stable wakefulness
        </h1>
        <p className="lead">
          Analysis code and statistical information supporting the manuscript.
          Materials will be updated as the study proceeds toward publication.
        </p>
        <p className="authors">
          Siqi Zhou, Takeshi Sawada, Hitoshi Okazaki <span>et al.</span>
        </p>
      </section>

      <section className="codeSection" id="code">
        <div className="sectionIntro">
          <p className="sectionLabel">Analysis code</p>
          <h2>Code used in the manuscript</h2>
          <p>
            The repository contains analysis scripts used to produce the
            reported results. File names and brief usage notes will be added as
            the code is consolidated.
          </p>
        </div>

        <div className="codeList">
          {codeItems.map((item) => {
            const content = (
              <>
                <div>
                  <h3>{item.title}</h3>
                  <p>{item.description}</p>
                </div>
                <span className="status">{item.status}</span>
              </>
            );

            return item.href ? (
              <a
                href={item.href}
                target="_blank"
                rel="noreferrer"
                key={item.title}
              >
                {content}
              </a>
            ) : (
              <div key={item.title}>{content}</div>
            );
          })}
        </div>
      </section>

      <section className="statistics" id="statistics">
        <div>
          <p className="sectionLabel">Statistical methods</p>
          <h2>A concise statistical record</h2>
        </div>
        <div className="statisticsCopy">
          <p>
            This section will provide the statistical tests, sample definitions,
            software, and reporting conventions used in the manuscript.
          </p>
          <dl>
            <div>
              <dt>Analysis software</dt>
              <dd>
                EEG analysis ·{" "}
                <a
                  href="https://github.com/tkssawada/SynC"
                  target="_blank"
                  rel="noreferrer"
                >
                  tkssawada/SynC ↗
                </a>
              </dd>
            </div>
            <div>
              <dt>Statistical tests</dt>
              <dd className="testDetails">
                <a className="detailLink" href="/statistical-tests">
                  Read test descriptions →
                </a>
                <a className="detailLink" href="/python-code">
                  View Python scripts →
                </a>
              </dd>
            </div>
          </dl>
          <p className="note">
            Figure-specific notes can be added later where they materially help
            readers reproduce an analysis.
          </p>
        </div>
      </section>

      <section className="citation" id="citation">
        <p className="sectionLabel">Citation and access</p>
        <div>
          <h2>Publication details will be added when available.</h2>
          <a
            className="repoLink"
            href="https://github.com/tkssawada/SynC"
            target="_blank"
            rel="noreferrer"
          >
            tkssawada/SynC on GitHub ↗
          </a>
        </div>
      </section>

      <footer>
        <span>SynC analysis resources</span>
        <span>Pre-publication draft</span>
      </footer>
    </main>
  );
}
