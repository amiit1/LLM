const API_BASE_URL = "http://127.0.0.1:8000/api/nlp";

const inputText = document.getElementById("inputText");
const statusText = document.getElementById("statusText");
const loadingBox = document.getElementById("loadingBox");
const resultPanel = document.getElementById("resultPanel");
const resultTitle = document.getElementById("resultTitle");
const resultContent = document.getElementById("resultContent");
const analyzeBtn = document.getElementById("analyzeBtn");
const sampleBtn = document.getElementById("sampleBtn");
const copyBtn = document.getElementById("copyBtn");

let latestResultPayload = null;

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
