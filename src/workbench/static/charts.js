// Small helpers around Chart.js: line charts with a linear x axis and {x, y} points,
// so series with different x values (train every N steps, val every epoch) share one chart.

const PALETTE = ['#2a6fdb', '#e8590c', '#2b8a3e', '#9c36b5', '#c92a2a'];

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function makeDataset(s, i) {
  const color = s.color || PALETTE[i % PALETTE.length];
  return {
    label: s.label, data: [], yAxisID: s.axis || 'y', key: s.key,
    borderColor: color, backgroundColor: color,
    borderDash: s.dash || [], borderWidth: 1.5,
    pointRadius: s.points ? 2.5 : 0, tension: 0,
  };
}

function lineChart(canvas, { xLabel = 'step', yLabel = '', series = [], y2 = null, yLog = false } = {}) {
  const grid = cssVar('--grid'), text = cssVar('--muted');
  const scales = {
    x: { type: 'linear', title: { display: true, text: xLabel, color: text },
         grid: { color: grid }, ticks: { color: text } },
    y: { type: yLog ? 'logarithmic' : 'linear',
         title: { display: !!yLabel, text: yLabel, color: text },
         grid: { color: grid }, ticks: { color: text } },
  };
  if (y2) {
    scales.y2 = { position: 'right', min: 0, max: 100, title: { display: true, text: y2, color: text },
                  grid: { drawOnChartArea: false }, ticks: { color: text } };
  }
  return new Chart(canvas, {
    type: 'line',
    data: { datasets: series.map(makeDataset) },
    options: {
      animation: false, parsing: false, normalized: true, spanGaps: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      plugins: { legend: { labels: { color: text, boxWidth: 12 } } },
      scales,
    },
  });
}

// Append points to dataset i; points is an array of {x, y}
function addPoints(chart, i, points) {
  if (points.length) chart.data.datasets[i].data.push(...points);
}

function fmt(v, digits = 4) {
  if (v === undefined || v === null) return '—';
  return typeof v === 'number' ? Number(v.toPrecision(digits)).toString() : String(v);
}

// Index of the dataset with this key; created (with props from `make(i)`) if it does not exist yet
function ensureDataset(chart, key, make) {
  const datasets = chart.data.datasets;
  let i = datasets.findIndex(d => d.key === key);
  if (i < 0) {
    i = datasets.length;
    datasets.push(makeDataset({ key, ...make(i) }, i));
  }
  return i;
}

// Points of every weight_delta_<lag> key, as [{lag, points}] sorted by lag
function weightDeltaSeries(records) {
  const byLag = {};
  for (const r of records) {
    if (r.step === undefined) continue;
    for (const k in r) {
      const m = /^weight_delta_(\d+)$/.exec(k);
      if (m) (byLag[m[1]] ||= []).push({ x: r.step, y: r[k] });
    }
  }
  return Object.keys(byLag).map(Number).sort((a, b) => a - b)
    .map(lag => ({ lag, points: byLag[lag] }));
}

// Line search (optimizer sgd_ls): mean shortening factor per logged window, accepted step / start
// step (= 0.5^halvings). ls_factor if logged, else ls_step / lr (runs before ls_factor existed).
function shorteningSeries(records) {
  return records
    .filter(r => r.step !== undefined && (r.ls_factor !== undefined || (r.ls_step !== undefined && r.lr)))
    .map(r => ({ x: r.step, y: r.ls_factor ?? r.ls_step / r.lr }));
}
