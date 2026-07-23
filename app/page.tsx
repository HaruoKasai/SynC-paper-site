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
    status: "tkssawada/SynC ↗",
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
          <a href="/statistical-tests">Statistics</a>
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
          <div className="statisticalItem">
            <div>
              <h3>Statistical analysis</h3>
              <p>
                Descriptions of the statistical tests and the corresponding
                Python scripts used in the manuscript.
              </p>
            </div>
            <div className="codeActions">
              <a href="/statistical-tests">Test descriptions →</a>
              <a href="/python-code">Python scripts →</a>
            </div>
          </div>
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
