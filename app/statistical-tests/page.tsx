import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Statistical test descriptions | SynC",
  description:
    "Descriptions of statistical procedures used in the SynC manuscript.",
};

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
        <p className="eyebrow">Figure 5 · Statistical methods</p>
        <h1>Population-activity and permutation analysis</h1>
        <p className="lead">
          This analysis quantifies neuronal activity rate, pairwise population
          coordination, and population dimensionality across behavioural states
          and time periods surrounding A/C administration.
        </p>
      </header>

      <article
        className="methodArticle"
        id="fig5-permutation"
        aria-label="Figure 5 permutation analysis"
      >
        <section>
          <p className="sectionLabel">Input data</p>
          <h2>Ten mice across two behavioural states</h2>
          <p>
            Each row of the processed input table contains one mouse-level
            estimate for a specified metric, behavioural state, and time window.
            The states were classified as <strong>Immobile</strong> or{" "}
            <strong>Mobile</strong>.
          </p>
          <div className="methodGrid">
            <div>
              <span>Before A/C</span>
              <strong>−45 to 0 min</strong>
            </div>
            <div>
              <span>After A/C</span>
              <strong>0 to 45 min</strong>
            </div>
            <div>
              <span>1 h after A/C</span>
              <strong>45 to 120 min</strong>
            </div>
          </div>
          <h3>Reported metrics</h3>
          <ol>
            <li>Firing rate</li>
            <li>Mean pairwise Spearman correlation</li>
            <li>Normalised participation ratio (PR norm)</li>
            <li>
              Normalised number of principal components explaining 50% of the
              variance (PC50 norm)
            </li>
          </ol>
          <p>
            Mean pairwise Spearman correlation is the signed mean of pairwise
            coefficients across the recorded neuronal population. PR norm
            represents effective dimensionality calculated from the covariance
            eigenvalue spectrum and normalised by neuronal number. PC50 norm is
            the number of principal components required to explain 50% of
            population variance, also normalised by neuronal number.
          </p>
        </section>

        <section>
          <p className="sectionLabel">Statistical analysis</p>
          <h2>Two-sided paired exact sign-flip permutation test</h2>
          <p>
            For each mouse, the paired difference was calculated as the
            comparison value minus its Before value. The test statistic was the
            mean within-mouse difference.
          </p>
          <div className="equation">
            d<sub>i</sub> = Y<sub>i, comparison</sub> − Y<sub>i, Before</sub>
          </div>
          <div className="equation">
            T<sub>obs</sub> = (1/n) Σ d<sub>i</sub>
          </div>
          <p>
            Under the null hypothesis, the sign of each within-mouse difference
            was considered exchangeable. With n = 10, all 2¹⁰ = 1,024 possible
            sign assignments were enumerated. The exact two-sided P value was
            the proportion of permuted absolute statistics at least as large as
            the observed absolute statistic.
          </p>
          <h3>Comparisons</h3>
          <ol>
            <li>After A/C versus Before A/C</li>
            <li>1 h after A/C versus Before A/C</li>
          </ol>
          <p>
            Comparisons were performed separately for Immobile and Mobile
            periods using a prespecified fixed sequence. The second comparison
            was interpreted as confirmatory only when the first was
            significant. No additional multiplicity adjustment was applied
            within each fixed sequence; both nominal exact P values are
            reported for transparency.
          </p>
        </section>

        <section>
          <p className="sectionLabel">Outputs and files</p>
          <h2>Reproducible analysis package</h2>
          <p>
            The script generates a CSV results table, an Excel workbook with
            summary and individual-mouse values, plots of individual
            trajectories and mean ± s.e.m., and the exact permutation P values.
          </p>
          <div className="articleLinks">
            <a className="repoLink" href="/python-code#fig5-permutation">
              Python script and input data →
            </a>
            <a
              className="repoLink"
              href="/docs/README_Fig5_permutation.md"
              download
            >
              Download original README
            </a>
          </div>
        </section>
      </article>

      <footer>
        <span>SynC statistical methods</span>
        <a href="/">Back to analysis resources</a>
      </footer>
    </main>
  );
}
