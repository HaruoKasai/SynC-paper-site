import type { Metadata } from "next";

export const metadata: Metadata = {
  title:
    "SynC | Rapid associative spine enlargement and stable wakefulness",
  description:
    "A concise research site for SynC, a reversible synaptic chemogenetic perturbation used to test the online role of associative spine enlargement.",
};

const findings = [
  {
    number: "01",
    title: "A selective, reversible perturbation",
    text: "SynC acutely blocks activity-dependent dendritic spine enlargement while sparing baseline excitability, NMDA receptor-dependent responses, and spontaneous calcium activity.",
  },
  {
    number: "02",
    title: "Cognition and wakefulness become unstable",
    text: "Broad frontoparietal SynC activation impaired goal-directed behavior and intermittently induced State-C, an abrupt behavioral arrest distinct from conventional sleep.",
  },
  {
    number: "03",
    title: "Activity persists while cortical coupling falls",
    text: "Mean activity and wake-like gamma power were preserved during Interm-C, but pairwise neuronal correlations decreased and network dimensionality increased.",
  },
  {
    number: "04",
    title: "The structural target is confirmed in vivo",
    text: "Two-photon glutamate uncaging revealed seconds-scale associative enlargement in a write-permissive subset of neocortical spines. SynC blocked this response, which recovered within one hour.",
  },
];

export default function Home() {
  return (
    <main>
      <nav className="nav" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="SynC home">
          SynC
        </a>
        <div className="navLinks">
          <a href="#overview">Overview</a>
          <a href="#findings">Findings</a>
          <a href="#resources">Resources</a>
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="eyebrow">Research preview · Manuscript in preparation</div>
        <h1>
          Rapid associative spine enlargement is required for cognitive
          function and stable wakefulness
        </h1>
        <p className="lead">
          SynC provides an acute and reversible way to test whether rapid
          structural synaptic plasticity is required online—not only for
          learning, but for coherent cortical function during wakefulness.
        </p>
        <p className="authors">
          Siqi Zhou, Takeshi Sawada, Hitoshi Okazaki <span>et al.</span>
        </p>
        <div className="actions">
          <a
            className="button primary"
            href="https://github.com/tkssawada/SynC"
            target="_blank"
            rel="noreferrer"
          >
            View code on GitHub
          </a>
          <span className="button quiet" aria-label="Paper link coming later">
            Paper link upon publication
          </span>
        </div>
      </section>

      <section className="overview" id="overview">
        <p className="sectionLabel">Overview</p>
        <div className="overviewGrid">
          <h2>
            Wake-like firing is not enough. Cortical activity must remain
            functionally coupled.
          </h2>
          <div className="overviewCopy">
            <p>
              Associative synaptic plasticity is commonly treated as a
              mechanism for learning and memory. This study asks a more
              immediate question: is rapid spine enlargement also required from
              moment to moment for cognition and stable wakefulness?
            </p>
            <p>
              Using SynC in mice, we separated rapid structural plasticity from
              general synaptic transmission. The resulting cellular, circuit,
              and behavioral effects were reversible within approximately one
              hour.
            </p>
          </div>
        </div>
      </section>

      <section className="findings" id="findings">
        <p className="sectionLabel">Key findings</p>
        <div className="findingGrid">
          {findings.map((finding) => (
            <article className="finding" key={finding.number}>
              <span>{finding.number}</span>
              <h3>{finding.title}</h3>
              <p>{finding.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="takeaway">
        <p className="sectionLabel">Central conclusion</p>
        <blockquote>
          Rapid associative spine enlargement is an active online synaptic
          process required to maintain functional cortical coupling, cognitive
          function, and stable wakefulness.
        </blockquote>
      </section>

      <section className="resources" id="resources">
        <div>
          <p className="sectionLabel">Resources</p>
          <h2>Code now. Paper and data links when public.</h2>
        </div>
        <div className="resourceList">
          <a
            href="https://github.com/tkssawada/SynC"
            target="_blank"
            rel="noreferrer"
          >
            <span>Analysis code</span>
            <strong>tkssawada/SynC ↗</strong>
          </a>
          <div>
            <span>Manuscript</span>
            <strong>Link will be added upon publication</strong>
          </div>
          <div>
            <span>Correspondence</span>
            <strong>Haruo Kasai · The University of Tokyo</strong>
          </div>
        </div>
      </section>

      <footer>
        <span>SynC research project</span>
        <span>Pre-publication site draft</span>
      </footer>
    </main>
  );
}
