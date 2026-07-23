import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Statistical Python code | SynC",
  description:
    "Python scripts for statistical analyses used in the SynC manuscript.",
};

const scripts = [
  {
    id: "fig5-permutation",
    label: "Figure 5",
    name: "Paired exact sign-flip permutation test",
    file: "Fig5_Permutation_test.py",
    description:
      "Reproduces the n = 10 Fig. 5 analysis for firing rate, Spearman pairwise correlation, normalised participation ratio, and normalised PC50 in immobile and mobile periods.",
    input: "Fig5_permutation_input_N10.csv",
  },
];

export default function PythonCodePage() {
  return (
    <main>
      <nav className="nav" aria-label="Page navigation">
        <a className="brand" href="/">
          SynC
        </a>
        <div className="navLinks">
          <a href="/">Home</a>
          <a href="/statistical-tests">Test descriptions</a>
        </div>
      </nav>

      <header className="subpageHero">
        <p className="eyebrow">Reproducible analysis</p>
        <h1>Python code for statistical tests</h1>
        <p className="lead">
          Python scripts used for the statistical analyses will be collected
          here with short instructions and links to their corresponding
          methods.
        </p>
      </header>

      <section className="scriptCatalogue" aria-label="Python scripts">
        {scripts.map((script, index) => (
          <article className="scriptCard" id={script.id} key={script.id}>
            <div>
              <p className="sectionLabel">{script.label}</p>
              <h2>{script.name}</h2>
              <p>{script.description}</p>
              <a
                className="repoLink"
                href={`/statistical-tests#${script.id}`}
              >
                Read the test description →
              </a>
            </div>
            <div className="filePanel">
              <span>Files</span>
              <strong>{script.file}</strong>
              <strong>{script.input}</strong>
              <div className="fileLinks">
                <a href={`/code/${script.file}`} download>
                  Download Python
                </a>
                <a href={`/data/${script.input}`} download>
                  Download input CSV
                </a>
                <a href="/docs/README_Fig5_permutation.md" download>
                  Download README
                </a>
              </div>
            </div>
          </article>
        ))}
      </section>

      <section className="repositoryCallout">
        <div>
          <p className="sectionLabel">Repository</p>
          <h2>EEG analysis code</h2>
        </div>
        <a
          className="repoLink"
          href="https://github.com/tkssawada/SynC"
          target="_blank"
          rel="noreferrer"
        >
          tkssawada/SynC on GitHub ↗
        </a>
      </section>

      <footer>
        <span>SynC statistical code</span>
        <a href="/">Back to analysis resources</a>
      </footer>
    </main>
  );
}
