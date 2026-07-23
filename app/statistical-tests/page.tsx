import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Statistical test descriptions | SynC",
  description:
    "Descriptions of statistical procedures used in the SynC manuscript.",
};

const testSections = [
  {
    id: "fig5-permutation",
    label: "Figure 5",
    title: "Paired exact sign-flip permutation test",
    purpose:
      "To compare each mouse after SynC activation with its own baseline value for the Fig. 5 population-activity measures.",
    method:
      "A two-sided paired exact sign-flip permutation test was applied to the within-mouse differences. The test statistic was the mean paired difference. With n = 10 mice, all 2¹⁰ = 1,024 possible sign assignments were enumerated.",
    reporting:
      "Nominal exact P values are reported. Comparisons were After (0–45 min) versus Before (−45–0 min), and 1 h after (45–120 min) versus Before, separately for immobile and mobile periods.",
    data:
      "Four measures were tested: firing rate, Spearman pairwise correlation, normalised participation ratio, and normalised PC50.",
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
          manuscript, linked to its analysis code and input data.
        </p>
      </header>

      <section className="testCatalogue" aria-label="Statistical tests">
        {testSections.map((test, index) => (
          <article className="testCard" id={test.id} key={test.id}>
            <p className="sectionLabel">{test.label}</p>
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
              <div>
                <dt>Measures</dt>
                <dd>{test.data}</dd>
              </div>
            </dl>
            <a className="repoLink" href={`/python-code#${test.id}`}>
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
