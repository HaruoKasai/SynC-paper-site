import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Statistical test descriptions | SynC",
  description:
    "Descriptions of statistical procedures used in the SynC manuscript.",
};

const testSections = [
  {
    title: "Test name",
    purpose: "Describe the scientific question addressed by the test.",
    method:
      "Add the test assumptions, variables, grouping structure, and implementation details.",
    reporting:
      "Add the reported statistic, degrees of freedom where applicable, and P-value convention.",
  },
];

export default function StatisticalTestsPage() {
  return (
    <main>
      <nav className="nav" aria-label="Page navigation">
        <a className="brand" href="/">
          SynC
        </a>
        <div className="navLinks">
          <a href="/">Home</a>
          <a href="/python-code">Python code</a>
        </div>
      </nav>

      <header className="subpageHero">
        <p className="eyebrow">Statistical methods</p>
        <h1>Statistical test descriptions</h1>
        <p className="lead">
          A concise explanation of each statistical procedure used in the
          manuscript. Tests can be added here as the analyses are finalised.
        </p>
      </header>

      <section className="testCatalogue" aria-label="Statistical tests">
        {testSections.map((test, index) => (
          <article className="testCard" key={`${test.title}-${index}`}>
            <p className="sectionLabel">Test {index + 1}</p>
            <h2>{test.title}</h2>
            <dl>
              <div>
                <dt>Purpose</dt>
                <dd>{test.purpose}</dd>
              </div>
              <div>
                <dt>Method</dt>
                <dd>{test.method}</dd>
              </div>
              <div>
                <dt>Reporting</dt>
                <dd>{test.reporting}</dd>
              </div>
            </dl>
            <a className="repoLink" href="/python-code">
              Corresponding Python script →
            </a>
          </article>
        ))}
      </section>

      <footer>
        <span>SynC statistical methods</span>
        <a href="/">Back to analysis resources</a>
      </footer>
    </main>
  );
}
