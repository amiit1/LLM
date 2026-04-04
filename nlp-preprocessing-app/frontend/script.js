const API_BASE_URL = "http://127.0.0.1:8000/api/nlp";
const EMBEDDING_API_BASE_URL = "http://127.0.0.1:8000/api/embeddings";

const inputText = document.getElementById("inputText");
const statusText = document.getElementById("statusText");
const loadingBox = document.getElementById("loadingBox");
const resultPanel = document.getElementById("resultPanel");
const resultTitle = document.getElementById("resultTitle");
const resultContent = document.getElementById("resultContent");
const analyzeBtn = document.getElementById("analyzeBtn");
const sampleBtn = document.getElementById("sampleBtn");
const copyBtn = document.getElementById("copyBtn");

const embeddingInfoText = document.getElementById("embeddingInfoText");
const embeddingText = document.getElementById("embeddingText");
const embeddingTopK = document.getElementById("embeddingTopK");
const embeddingMethod = document.getElementById("embeddingMethod");
const embeddingLimit = document.getElementById("embeddingLimit");
const embeddingRunBtn = document.getElementById("embeddingRunBtn");
const embeddingVizBtn = document.getElementById("embeddingVizBtn");
const embeddingSampleCorpusBtn = document.getElementById("embeddingSampleCorpusBtn");
const embeddingStatus = document.getElementById("embeddingStatus");
const embeddingVectorMeta = document.getElementById("embeddingVectorMeta");
const embeddingVectorOutput = document.getElementById("embeddingVectorOutput");
const embeddingNeighborsOutput = document.getElementById("embeddingNeighborsOutput");
const embeddingPlotOutput = document.getElementById("embeddingPlotOutput");
const embeddingMethodBadge = document.getElementById("embeddingMethodBadge");

let latestResultPayload = null;
let cachedCorpus = [];

const sampleText =
  "Apple is planning to open a new office in Bengaluru in 2027. " +
  "John and Priya visited New York last summer for an AI conference.";

const entityColors = {
  PERSON: "bg-rose-100 text-rose-800 border-rose-200",
  ORG: "bg-sky-100 text-sky-800 border-sky-200",
  GPE: "bg-emerald-100 text-emerald-800 border-emerald-200",
  DATE: "bg-amber-100 text-amber-900 border-amber-200",
  MONEY: "bg-lime-100 text-lime-900 border-lime-200",
};

function setLoading(isLoading) {
  loadingBox.classList.toggle("hidden", !isLoading);
  loadingBox.classList.toggle("flex", isLoading);
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.className = isError
    ? "text-sm text-rose-600"
    : "text-sm text-slate-500";
}

function requireInputText() {
  const text = inputText.value.trim();
  if (!text) {
    setStatus("Please enter some text before running analysis.", true);
    inputText.focus();
    return null;
  }
  return text;
}

async function postToApi(endpoint, text) {
  const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }

  return payload;
}

async function fetchEmbeddingApi(path, options = {}) {
  const response = await fetch(`${EMBEDDING_API_BASE_URL}${path}`, options);
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Embedding request failed.");
  }

  return payload;
}

function setEmbeddingStatus(message, isError = false) {
  embeddingStatus.textContent = message;
  embeddingStatus.className = isError
    ? "text-sm text-rose-600"
    : "text-sm text-slate-500";
}

function sanitizeNumberInput(rawValue, min, max, fallback) {
  const parsed = Number.parseInt(rawValue, 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, parsed));
}

function formatProjectionMethod(method) {
  return method.toLowerCase() === "tsne" ? "t-SNE" : "PCA";
}

function renderEmbeddingVector(vector) {
  if (!vector.length) {
    return "<p class='text-slate-500'>No vector values available.</p>";
  }

  const preview = vector.slice(0, 14);
  const badges = preview
    .map(
      (value) =>
        `<span class='rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-xs font-semibold text-cyan-800'>${Number(
          value
        ).toFixed(4)}</span>`
    )
    .join(" ");

  return `
    <div class='flex flex-wrap gap-2'>${badges}</div>
    <p class='mt-3 text-xs text-slate-500'>Showing first ${preview.length} dimensions for readability.</p>
  `;
}

function renderEmbeddingNeighbors(neighbors) {
  if (!neighbors.length) {
    return "<p class='text-slate-500'>No similar corpus entries were found.</p>";
  }

  return neighbors
    .map(
      (item, index) => `
        <div class='flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2'>
          <div>
            <p class='font-medium text-slate-800'>${index + 1}. Doc ${item.doc_id}</p>
            <p class='text-xs text-slate-500'>${escapeHtml(item.text)}</p>
          </div>
          <span class='text-xs font-semibold text-slate-600'>cosine ${Number(item.score).toFixed(4)}</span>
        </div>
      `
    )
    .join("");
}

function renderProjection(points) {
  if (!points.length) {
    return "<p class='text-slate-500'>No projection points were returned.</p>";
  }

  const width = 860;
  const height = 360;
  const padding = 36;

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const content = points
    .map((point, idx) => {
      const normalizedX = (point.x - minX) / spanX;
      const normalizedY = (point.y - minY) / spanY;

      const x = padding + normalizedX * (width - padding * 2);
      const y = height - padding - normalizedY * (height - padding * 2);
      const radius = 3 + Math.min(3, point.importance * 35);
      const shouldShowLabel = idx < 24;

      return `
        <g>
          <circle cx='${x.toFixed(2)}' cy='${y.toFixed(2)}' r='${radius.toFixed(
            2
          )}' fill='#0284c7' opacity='0.72'>
            <title>${escapeHtml(point.label)}: ${escapeHtml(point.text)}</title>
          </circle>
          ${
            shouldShowLabel
              ? `<text x='${(x + 5).toFixed(2)}' y='${(y - 5).toFixed(2)}' class='plot-label'>${escapeHtml(
                  point.label
                )}</text>`
              : ""
          }
        </g>
      `;
    })
    .join("");

  return `
    <svg viewBox='0 0 ${width} ${height}' class='h-[360px] w-full rounded-xl border border-slate-200 bg-white'>
      <line x1='${padding}' y1='${height - padding}' x2='${width - padding}' y2='${height - padding}' stroke='#cbd5e1' />
      <line x1='${padding}' y1='${padding}' x2='${padding}' y2='${height - padding}' stroke='#cbd5e1' />
      ${content}
    </svg>
  `;
}

async function loadEmbeddingInfo() {
  try {
    const payload = await fetchEmbeddingApi("/info");
    embeddingInfoText.textContent = `${payload.technique} | corpus docs: ${payload.corpus_size} | vocabulary: ${payload.vocabulary_size}`;
  } catch (error) {
    embeddingInfoText.textContent = "Embedding metadata could not be loaded.";
  }
}

async function loadEmbeddingCorpus() {
  try {
    const payload = await fetchEmbeddingApi("/corpus?limit=80");
    cachedCorpus = payload.documents || [];
  } catch (error) {
    setEmbeddingStatus("Could not fetch corpus entries.", true);
  }
}

async function runEmbeddingLookup() {
  const text = embeddingText.value.trim();
  if (!text) {
    setEmbeddingStatus("Please enter corpus-related text to retrieve embedding.", true);
    embeddingText.focus();
    return;
  }

  const topK = sanitizeNumberInput(embeddingTopK.value, 1, 15, 5);
  embeddingTopK.value = String(topK);

  try {
    setLoading(true);
    setEmbeddingStatus("Fetching corpus embedding and nearest entries...");

    const payload = await fetchEmbeddingApi("/vector", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, top_k: topK }),
    });

    embeddingVectorMeta.textContent = `Corpus query embedded | dimensions: ${payload.dimension}`;
    embeddingVectorOutput.innerHTML = renderEmbeddingVector(payload.vector || []);
    embeddingNeighborsOutput.innerHTML = renderEmbeddingNeighbors(payload.neighbors || []);
    latestResultPayload = payload;
    setEmbeddingStatus("Corpus embedding lookup completed.");
  } catch (error) {
    setEmbeddingStatus(`Error: ${error.message}`, true);
  } finally {
    setLoading(false);
  }
}

async function runProjection() {
  const method = embeddingMethod.value;
  const limit = sanitizeNumberInput(embeddingLimit.value, 10, 120, 40);
  embeddingLimit.value = String(limit);

  try {
    setLoading(true);
    setEmbeddingStatus("Computing 2D projection...");

    const payload = await fetchEmbeddingApi(`/visualize?method=${method}&limit=${limit}`);
    embeddingMethodBadge.textContent = formatProjectionMethod(payload.method);
    embeddingPlotOutput.innerHTML = renderProjection(payload.points || []);
    latestResultPayload = payload;
    setEmbeddingStatus("Projection completed.");
  } catch (error) {
    setEmbeddingStatus(`Error: ${error.message}`, true);
  } finally {
    setLoading(false);
  }
}

function renderList(items) {
  if (!items.length) {
    return "<p class='text-slate-500'>No output returned.</p>";
  }

  const badges = items
    .map(
      (item) =>
        `<span class='rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-700'>${escapeHtml(
          String(item)
        )}</span>`
    )
    .join(" ");

  return `<div class='flex flex-wrap gap-2'>${badges}</div>`;
}

function renderPosTable(posTags) {
  if (!posTags.length) {
    return "<p class='text-slate-500'>No POS tags found.</p>";
  }

  const rows = posTags
    .map(
      (item) => `
      <tr class='border-b border-slate-100'>
        <td class='px-3 py-2 font-medium text-slate-800'>${escapeHtml(item.token)}</td>
        <td class='px-3 py-2'>${escapeHtml(item.pos)}</td>
        <td class='px-3 py-2'>${escapeHtml(item.tag)}</td>
        <td class='px-3 py-2 text-slate-500'>${escapeHtml(item.description || "-")}</td>
      </tr>
    `
    )
    .join("");

  return `
    <div class='overflow-x-auto'>
      <table class='w-full min-w-[600px] text-left text-xs sm:text-sm'>
        <thead>
          <tr class='bg-slate-100 text-slate-700'>
            <th class='px-3 py-2'>Token</th>
            <th class='px-3 py-2'>POS</th>
            <th class='px-3 py-2'>Tag</th>
            <th class='px-3 py-2'>Description</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderEntities(entities) {
  if (!entities.length) {
    return "<p class='text-slate-500'>No named entities found.</p>";
  }

  const tags = entities
    .map((entity) => {
      const colorClass = entityColors[entity.label] || "bg-violet-100 text-violet-800 border-violet-200";
      return `
        <div class='rounded-xl border px-3 py-2 ${colorClass}'>
          <p class='text-sm font-semibold'>${escapeHtml(entity.text)}</p>
          <p class='text-xs'>${escapeHtml(entity.label)} - ${escapeHtml(entity.description || "-")}</p>
        </div>
      `;
    })
    .join("");

  return `<div class='grid gap-2 sm:grid-cols-2'>${tags}</div>`;
}

function renderSingleResult(endpoint, payload) {
  if (endpoint === "tokenize") {
    return renderList(payload.tokens || []);
  }
  if (endpoint === "lemmatize") {
    return renderList(payload.lemmas || []);
  }
  if (endpoint === "stem") {
    return renderList(payload.stems || []);
  }
  if (endpoint === "pos-tag") {
    return renderPosTable(payload.pos_tags || []);
  }
  if (endpoint === "ner") {
    return renderEntities(payload.entities || []);
  }

  return "<p class='text-slate-500'>Unsupported endpoint.</p>";
}

function renderAnalyzeAll(payload) {
  return `
    <div class='space-y-6'>
      <div>
        <h3 class='mb-2 text-base font-semibold text-slate-800'>Tokens</h3>
        ${renderList(payload.tokens || [])}
      </div>

      <div>
        <h3 class='mb-2 text-base font-semibold text-slate-800'>Lemmas</h3>
        ${renderList(payload.lemmas || [])}
      </div>

      <div>
        <h3 class='mb-2 text-base font-semibold text-slate-800'>Stems</h3>
        ${renderList(payload.stems || [])}
      </div>

      <div>
        <h3 class='mb-2 text-base font-semibold text-slate-800'>POS Tags</h3>
        ${renderPosTable(payload.pos_tags || [])}
      </div>

      <div>
        <h3 class='mb-2 text-base font-semibold text-slate-800'>Named Entities</h3>
        ${renderEntities(payload.entities || [])}
      </div>
    </div>
  `;
}

function displayResult(title, html, payload) {
  resultTitle.textContent = title;
  resultContent.innerHTML = html;
  resultPanel.classList.remove("hidden");
  resultPanel.classList.add("fade-in");
  latestResultPayload = payload;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function runSingleTask(endpoint) {
  const text = requireInputText();
  if (!text) {
    return;
  }

  try {
    setLoading(true);
    setStatus(`Running ${endpoint}...`);
    const payload = await postToApi(endpoint, text);
    const html = renderSingleResult(endpoint, payload);
    displayResult(`Result: ${endpoint}`, html, payload);
    setStatus(`Completed ${endpoint}.`);
  } catch (error) {
    setStatus(`Error: ${error.message}`, true);
  } finally {
    setLoading(false);
  }
}

async function runAnalyzeAll() {
  const text = requireInputText();
  if (!text) {
    return;
  }

  try {
    setLoading(true);
    setStatus("Running complete analysis...");
    const payload = await postToApi("analyze", text);
    const html = renderAnalyzeAll(payload);
    displayResult("Result: Analyze All", html, payload);
    setStatus("Completed full NLP analysis.");
  } catch (error) {
    setStatus(`Error: ${error.message}`, true);
  } finally {
    setLoading(false);
  }
}

document.querySelectorAll("[data-endpoint]").forEach((button) => {
  button.addEventListener("click", () => runSingleTask(button.dataset.endpoint));
});

analyzeBtn.addEventListener("click", runAnalyzeAll);

sampleBtn.addEventListener("click", () => {
  inputText.value = sampleText;
  setStatus("Sample text loaded. Click any action button.");
});

embeddingRunBtn.addEventListener("click", runEmbeddingLookup);
embeddingVizBtn.addEventListener("click", runProjection);

embeddingSampleCorpusBtn.addEventListener("click", () => {
  if (!cachedCorpus.length) {
    setEmbeddingStatus("Corpus is still loading. Please try again.", true);
    return;
  }

  const randomIndex = Math.floor(Math.random() * cachedCorpus.length);
  embeddingText.value = cachedCorpus[randomIndex].text;
  setEmbeddingStatus("Sample corpus text loaded. Click Get Embedding.");
});

copyBtn.addEventListener("click", async () => {
  if (!latestResultPayload) {
    setStatus("No result available to copy.", true);
    return;
  }

  try {
    await navigator.clipboard.writeText(JSON.stringify(latestResultPayload, null, 2));
    setStatus("Result copied to clipboard.");
  } catch (error) {
    setStatus("Clipboard access failed in this browser context.", true);
  }
});

loadEmbeddingInfo();
loadEmbeddingCorpus();
