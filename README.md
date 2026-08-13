# SynC paper analysis resources

This repository publishes statistical methods, frozen source tables, and
reproducible Python scripts supporting the SynC manuscript.

## Published analyses

- **Figure 5:** paired exact sign-flip permutation analysis of population
  activity.
- **Figure 6g:** 40-80-s mean spine-volume change analysed from equally
  weighted FOV means with a two-sided heteroscedastic Normal parametric
  bootstrap and mouse-level random intercept.
- **Figure 6h / Extended Data Figure 10:** Normal-Exponential mixture model,
  condition-specific mixture fractions, common-prior FOV-mean posterior-score
  sensitivity analysis, and distributional audit.

The website provides method summaries under `/statistical-tests` and direct
downloads under `/python-code`. Analysis-specific provenance and execution
instructions are stored in `public/docs/`.

## Local development

Node.js 22.13 or newer is required.

```bash
npm install
npm run dev
```

Validate a production build and the rendered routes with:

```bash
npm test
```

## Reproducibility files

- `public/code/`: Python analysis and audit scripts.
- `public/data/`: public, path-sanitised input and reported-result CSVs.
- `public/docs/`: methods, model definitions, seeds, replicate counts, source
  hashes, and adopted fixed-run results.
