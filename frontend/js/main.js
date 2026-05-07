function renderList(elementId, rows, renderRow) {
  const root = document.getElementById(elementId);
  if (!root) return;
  if (!rows || rows.length === 0) {
    root.innerHTML = '<div class="small">暂无数据</div>';
    return;
  }
  root.innerHTML = rows.map(renderRow).join("");
}
