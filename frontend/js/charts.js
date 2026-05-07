function renderSimpleBar(containerId, data) {
  const root = document.getElementById(containerId);
  if (!root) return;
  const max = Math.max(...data.map(x => x.value), 1);
  root.innerHTML = data
    .map(
      x => `
      <div style="margin-bottom:8px;">
        <div class="small">${x.label}: ${x.value}</div>
        <div style="height:8px;background:#e5e7eb;border-radius:999px;overflow:hidden;">
          <div style="height:8px;background:#0f766e;width:${(x.value / max) * 100}%"></div>
        </div>
      </div>`
    )
    .join("");
}
