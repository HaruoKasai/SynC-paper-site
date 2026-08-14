import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import test from "node:test";

const root = new URL("../", import.meta.url);
const builtAssetRoot = fileURLToPath(new URL("../dist/client/", import.meta.url));
let workerPromise;

async function worker() {
  workerPromise ??= import(
    new URL(`../dist/server/index.js?test=${process.pid}-${Date.now()}`, import.meta.url)
      .href
  ).then((module) => module.default);
  return workerPromise;
}

async function serveBuiltAsset(request) {
  const requestUrl =
    typeof request === "string"
      ? request
      : request instanceof URL
        ? request.href
        : request.url;
  const pathname = decodeURIComponent(new URL(requestUrl).pathname);
  const filePath = join(builtAssetRoot, pathname.replace(/^\/+/, ""));
  try {
    const body = await readFile(filePath);
    const extension = filePath.split(".").pop()?.toLowerCase();
    const contentTypes = {
      css: "text/css; charset=utf-8",
      csv: "text/csv; charset=utf-8",
      js: "text/javascript; charset=utf-8",
      json: "application/json; charset=utf-8",
      md: "text/markdown; charset=utf-8",
      py: "text/x-python; charset=utf-8",
      zip: "application/zip",
    };
    return new Response(body, {
      status: 200,
      headers: { "content-type": contentTypes[extension] ?? "application/octet-stream" },
    });
  } catch {
    return new Response("Not found", {
      status: 404,
      headers: { "x-test-asset-path": filePath },
    });
  }
}

async function fetchApp(pathname, accept = "text/html") {
  const app = await worker();
  return app.fetch(
    new Request(`http://localhost${pathname}`, {
      headers: { accept },
    }),
    {
      ASSETS: { fetch: serveBuiltAsset },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

async function render(pathname) {
  return fetchApp(pathname, "text/html");
}

function parseSimpleCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) =>
    Object.fromEntries(headers.map((header, index) => [header, line.split(",")[index]])),
  );
}

test("server-renders the analysis resource routes", async () => {
  const cases = [
    ["/", /Rapid associative spine enlargement/],
    ["/statistical-tests", /Two-sided FOV-level parametric bootstrap/],
    ["/python-code", /EEG\/EMG preprocessing and spectral analysis/],
  ];

  for (const [pathname, expected] of cases) {
    const response = await render(pathname);
    assert.equal(response.status, 200, pathname);
    assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
    const html = await response.text();
    assert.match(html, expected);
    assert.doesNotMatch(html, /Your site is taking shape|codex-preview/);
  }
});

test("publishes the Fig. 4c EEG analysis and synthetic workflow demo", async () => {
  const paths = [
    "public/code/Fig4c_EEG_analysis.py",
    "public/data/Fig4c_EEG_demo.npz",
    "public/docs/README_Fig4c_EEG.md",
  ];
  await Promise.all(paths.map((path) => access(new URL(path, root))));

  const [homePage, codePage, analysisCode, readme] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/python-code/page.tsx", root), "utf8"),
    readFile(new URL("public/code/Fig4c_EEG_analysis.py", root), "utf8"),
    readFile(new URL("public/docs/README_Fig4c_EEG.md", root), "utf8"),
  ]);

  assert.match(homePage, /\/python-code#fig4c-eeg/);
  assert.doesNotMatch(homePage, /tkssawada\/SynC/);
  assert.match(codePage, /Fig4c_EEG_analysis\.py/);
  assert.match(codePage, /Fig4c_EEG_demo\.npz/);
  assert.match(codePage, /2,000-Hz synthetic NPZ/);
  assert.doesNotMatch(codePage, /tkssawada\/SynC/);
  assert.match(analysisCode, /BlackrockIO/);
  assert.match(analysisCode, /f\.endswith\(\("\.ns3", "\.ns2"\)\)/);
  assert.match(readme, /raw Blackrock `\.ns2`\/`\.ns3` recordings/);
  assert.match(readme, /synthetic demonstration data/);
});

test("publishes the frozen Fig. 6 and Extended Data Fig. 10 package", async () => {
  const paths = [
    "public/code/Fig6_FOV_parametric_bootstrap.py",
    "public/code/ExFig10_mixture_audit.py",
    "public/data/Fig6_ExFig10_FOV_input.csv",
    "public/data/Fig6_ExFig10_spine_input.csv",
    "public/data/Fig6_ExFig10_mixture_parameters.csv",
    "public/data/Fig6_ExFig10_cohort_counts.csv",
    "public/data/Fig6_ExFig10_reported_tests.csv",
    "public/docs/README_Fig6_ExFig10.md",
  ];
  await Promise.all(paths.map((path) => access(new URL(path, root))));

  const [fovCsv, spineCsv, testCsv, methodsPage, codePage, methodsReadme, parameters] =
    await Promise.all([
      readFile(new URL("public/data/Fig6_ExFig10_FOV_input.csv", root), "utf8"),
      readFile(new URL("public/data/Fig6_ExFig10_spine_input.csv", root), "utf8"),
      readFile(new URL("public/data/Fig6_ExFig10_reported_tests.csv", root), "utf8"),
      readFile(new URL("app/statistical-tests/page.tsx", root), "utf8"),
      readFile(new URL("app/python-code/page.tsx", root), "utf8"),
      readFile(new URL("public/docs/README_Fig6_ExFig10.md", root), "utf8"),
      readFile(new URL("public/data/Fig6_ExFig10_mixture_parameters.csv", root), "utf8"),
    ]);

  assert.equal(
    fovCsv.split(/\r?\n/, 1)[0],
    "group,mouse_id,fov_id,n_spines,mean_delta_v_40_80_percent,permissive_fraction",
  );
  assert.equal(
    spineCsv.split(/\r?\n/, 1)[0],
    "group,role,mouse_id,fov_id,spine_id,corrected_delta_v_40_80_percent,posterior_permissive",
  );
  assert.equal(
    testCsv.split(/\r?\n/, 1)[0],
    "metric,contrast_id,alternative,effect,display_p,seed,repetitions,inference_role",
  );
  assert.doesNotMatch(spineCsv, /[A-Z]:\\Users\\/i);
  assert.equal(spineCsv.trim().split(/\r?\n/).length - 1, 1088);
  assert.equal((spineCsv.match(/,stim,/g) ?? []).length, 565);
  assert.equal((spineCsv.match(/,neighbor,/g) ?? []).length, 523);
  assert.match(testCsv, /sync_before_vs_0_60,two-sided,6\.286/);
  assert.match(testCsv, /0\.048795120487951205/);
  assert.match(testCsv, /0\.0058994100589941/);
  assert.match(testCsv, /0\.762023797620238/);
  assert.match(testCsv, /0\.48135186481351866/);
  assert.match(testCsv, /0\.004899510048995101/);
  assert.match(testCsv, /0\.0143985601439856/);
  assert.match(testCsv, /0\.7952204779522047/);
  assert.match(testCsv, /0\.8823117688231177/);
  assert.doesNotMatch(testCsv, /wt_vs_sync_before/);
  assert.doesNotMatch(testCsv, /sync_before_vs_60_180/);
  assert.match(methodsPage, /mouse-level random intercept/);
  assert.match(
    methodsPage,
    /Individual spine\s+responses were not regenerated/,
  );
  assert.match(methodsPage, /Percentograms are used only for visualisation/);
  assert.doesNotMatch(methodsPage, /WT versus SynC -A\/C uses/);
  assert.doesNotMatch(methodsPage, /0\.3810|0\.1905/);
  assert.match(codePage, /Fig6_FOV_parametric_bootstrap\.py/);
  assert.match(parameters, /pi_WT/);
  assert.match(parameters, /common_pi_sensitivity/);
  assert.match(methodsReadme, /WT rows and the frozen `pi_WT` parameter remain/);
  assert.doesNotMatch(methodsReadme, /100,000|0\.0517/);
});

test("publishes only stable opaque analysis identifiers", async () => {
  const [fovText, spineText] = await Promise.all([
    readFile(new URL("public/data/Fig6_ExFig10_FOV_input.csv", root), "utf8"),
    readFile(new URL("public/data/Fig6_ExFig10_spine_input.csv", root), "utf8"),
  ]);
  const fovRows = parseSimpleCsv(fovText);
  const spineRows = parseSimpleCsv(spineText);

  assert.equal(new Set(fovRows.map((row) => row.mouse_id)).size, 11);
  assert.equal(new Set(fovRows.map((row) => row.fov_id)).size, 158);
  assert.ok(
    fovRows.every(
      (row) =>
        /^M6-\d{3}$/.test(row.mouse_id) &&
        /^FOV-\d{3}$/.test(row.fov_id),
    ),
  );

  const publishedFovs = new Set(fovRows.map((row) => row.fov_id));
  assert.equal(new Set(spineRows.map((row) => row.spine_id)).size, 1088);
  assert.ok(
    spineRows.every(
      (row) =>
        /^M6-\d{3}$/.test(row.mouse_id) &&
        publishedFovs.has(row.fov_id) &&
        /^SP-\d{4}$/.test(row.spine_id),
    ),
  );
});

test("emits built styles and public downloads for the Sites asset binding", async () => {
  const home = await render("/");
  const html = await home.text();
  const stylesheet = html.match(/href="([^"]+\.css)"/)?.[1];
  assert.ok(stylesheet, "rendered HTML should link a stylesheet");

  const requests = [
    [stylesheet, "text/css"],
    ["/code/Fig4c_EEG_analysis.py", "text/x-python"],
    ["/data/Fig4c_EEG_demo.npz", "application/octet-stream"],
    ["/docs/README_Fig4c_EEG.md", "text/markdown"],
    ["/code/Fig6_FOV_parametric_bootstrap.py", "text/x-python"],
    ["/data/Fig6_ExFig10_FOV_input.csv", "text/csv"],
    ["/docs/README_Fig6_ExFig10.md", "text/markdown"],
  ];
  const responses = await Promise.all(
    requests.map(([pathname, accept]) =>
      serveBuiltAsset(
        new Request(`http://localhost${pathname}`, {
          headers: { accept },
        }),
      ),
    ),
  );

  for (const [index, response] of responses.entries()) {
    assert.equal(
      response.status,
      200,
      `${requests[index][0]} -> ${response.headers.get("x-test-asset-path") ?? "no asset path"}`,
    );
  }
});

test("does not publish local drive or network paths", async () => {
  const paths = [
    "public/code/Fig4c_EEG_analysis.py",
    "public/docs/README_Fig4c_EEG.md",
    "public/code/Fig6_FOV_parametric_bootstrap.py",
    "public/code/ExFig10_mixture_audit.py",
    "public/data/Fig6_ExFig10_FOV_input.csv",
    "public/data/Fig6_ExFig10_spine_input.csv",
    "public/data/Fig6_ExFig10_mixture_parameters.csv",
    "public/data/Fig6_ExFig10_cohort_counts.csv",
    "public/data/Fig6_ExFig10_reported_tests.csv",
    "public/docs/README_Fig6_ExFig10.md",
  ];

  for (const path of paths) {
    const content = await readFile(new URL(path, root), "utf8");
    assert.doesNotMatch(content, /\b[A-Za-z]:\\+/);
    assert.doesNotMatch(content, /\\\\[A-Za-z0-9_.-]+\\/);
  }
});
