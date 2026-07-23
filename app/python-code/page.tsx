import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Statistical Python code | SynC",
  description:
    "Python scripts for statistical analyses used in the SynC manuscript.",
};

const scripts = [
  {
    name: "Statistical test script",
    file: "Python file to be added",
    description:
      "This entry will contain the script, required inputs, expected output, and a link to the related test description.",
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
          <article className="scriptCard" key={`${script.name}-${index}`}>
            <div>
              <p className="sectionLabel">Script {index + 1}</p>
              <h2>{script.name}</h2>
              <p>{script.description}</p>
            </div>
            <div className="filePanel">
              <span>File</span>
              <strong>{script.file}</strong>
              <span className="status">In preparation</span>
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
