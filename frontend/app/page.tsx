"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Dataset = {
  name: string;
  path: string;
};

type BacktestPoint = {
  date: string;
  equity: number;
  drawdown: number;
  position: number;
};

type BacktestTrade = {
  date: string;
  action: string;
  price: number;
  shares: number;
};

type BacktestResult = {
  asset_name: string;
  strategy_name: string;
  metrics: Record<string, string | number>;
  history: BacktestPoint[];
  trades: BacktestTrade[];
};

type MlMetadata = {
  requested_name?: string;
  model_name: string;
  accuracy: number;
  validation_mode: string;
  test_start_index: number;
  num_predictions: number;
};

type ApiResponse = {
  asset: string;
  strategy?: string;
  strategies?: string[];
  ml_metadata: MlMetadata | MlMetadata[] | null;
  results: BacktestResult[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const STRATEGY_OPTIONS = [
  { value: "ma", label: "Moving Average" },
  { value: "rsi", label: "RSI" },
  { value: "ml", label: "ML Strategy" },
  { value: "buy_hold", label: "Buy and Hold" },
] as const;
const SERIES_COLORS = ["#0f766e", "#2563eb", "#d97706", "#b91c1c", "#7c3aed"];

export default function Page() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState("");
  const [strategy, setStrategy] = useState("ml");
  const [mlModel, setMlModel] = useState("logistic");
  const [compareMode, setCompareMode] = useState(true);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>(["ma", "rsi", "ml", "buy_hold"]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ApiResponse | null>(null);

  const loadDatasets = async (preferredDataset?: string) => {
    setDatasetsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/datasets`);
      if (!response.ok) {
        throw new Error("Dataset request failed.");
      }
      const payload = (await response.json()) as { datasets: Dataset[] };
      setDatasets(payload.datasets);

      if (payload.datasets.length === 0) {
        setDataset("");
        setError("No datasets found yet. Upload a CSV or add one to the data folder.");
        return;
      }

      setError("");
      const nextDataset =
        preferredDataset && payload.datasets.some((item) => item.name === preferredDataset)
          ? preferredDataset
          : payload.datasets[0].name;
      setDataset(nextDataset);
    } catch {
      setError(
        "Could not load datasets from the backend. Make sure FastAPI is running on http://localhost:8000.",
      );
    } finally {
      setDatasetsLoading(false);
    }
  };

  useEffect(() => {
    void loadDatasets();
  }, []);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const requestBody = compareMode
        ? {
            dataset,
            strategies: selectedStrategies,
            ml_model: mlModel,
            validation_mode: "walk_forward",
            min_train_size: 252,
            step_size: 21,
          }
        : {
            dataset,
            strategy,
            compare_buy_hold: true,
            ml_model: mlModel,
            validation_mode: "walk_forward",
            min_train_size: 252,
            step_size: 21,
          };

      const response = await fetch(`${API_BASE}/api/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const payload = (await response.json()) as { detail?: string };
        throw new Error(payload.detail ?? "Backtest request failed.");
      }

      const payload = (await response.json()) as ApiResponse;
      setResult(payload);
    } catch (submissionError) {
      setError(submissionError instanceof Error ? submissionError.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const input = form.elements.namedItem("datasetFile") as HTMLInputElement | null;
    const file = input?.files?.[0];

    if (!file) {
      setError("Choose a CSV file before uploading.");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = (await response.json()) as { detail?: string };
        throw new Error(payload.detail ?? "Upload failed.");
      }

      const payload = (await response.json()) as { dataset: Dataset; datasets: Dataset[] };
      setDatasets(payload.datasets);
      setDataset(payload.dataset.name);
      form.reset();
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const chartSeries = useMemo(
    () =>
      result?.results.map((item, index) => ({
        label: formatStrategy(item.strategy_name),
        values: item.history.map((point) => point.equity),
        color: SERIES_COLORS[index % SERIES_COLORS.length],
      })) ?? [],
    [result],
  );

  const drawdownSeries = useMemo(
    () =>
      result?.results.map((item, index) => ({
        label: formatStrategy(item.strategy_name),
        values: item.history.map((point) => point.drawdown),
        color: SERIES_COLORS[index % SERIES_COLORS.length],
      })) ?? [],
    [result],
  );

  const timeline = result?.results[0]?.history.map((point) => point.date) ?? [];
  const mlMetadataList = normalizeMlMetadata(result?.ml_metadata ?? null);

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Backtesting Dashboard</p>
          <h1>Compare trading strategies through one browser interface.</h1>
          <p className="hero-copy">
            Run one strategy or compare several side by side using the same Python backtesting engine,
            then inspect metrics, equity overlays, and drawdown behavior.
          </p>
        </div>

        <form className="control-panel" onSubmit={handleSubmit}>
          <label>
            Dataset
            <select
              value={dataset}
              onChange={(event) => setDataset(event.target.value)}
              disabled={datasetsLoading || datasets.length === 0}
            >
              {datasets.length === 0 ? (
                <option value="">{datasetsLoading ? "Loading datasets..." : "No datasets available"}</option>
              ) : null}
              {datasets.map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>

          <div className="toggle-row">
            <button
              type="button"
              className={`toggle-chip ${compareMode ? "active" : ""}`}
              onClick={() => setCompareMode(true)}
            >
              Compare Strategies
            </button>
            <button
              type="button"
              className={`toggle-chip ${!compareMode ? "active" : ""}`}
              onClick={() => setCompareMode(false)}
            >
              Single Strategy
            </button>
          </div>

          {compareMode ? (
            <fieldset className="strategy-group">
              <legend>Strategies to compare</legend>
              {STRATEGY_OPTIONS.map((item) => (
                <label key={item.value} className="check-row">
                  <input
                    type="checkbox"
                    checked={selectedStrategies.includes(item.value)}
                    onChange={() => setSelectedStrategies((current) => toggleStrategy(current, item.value))}
                  />
                  <span>{item.label}</span>
                </label>
              ))}
            </fieldset>
          ) : (
            <label>
              Strategy
              <select value={strategy} onChange={(event) => setStrategy(event.target.value)}>
                {STRATEGY_OPTIONS.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          )}

          <label>
            ML Model
            <select
              value={mlModel}
              onChange={(event) => setMlModel(event.target.value)}
              disabled={!compareMode && strategy !== "ml" && !selectedStrategies.includes("ml")}
            >
              <option value="logistic">Logistic Regression</option>
              <option value="random_forest">Random Forest</option>
            </select>
          </label>

          <div className="button-row">
            <button
              type="button"
              className="secondary-button"
              onClick={() => void loadDatasets(dataset)}
              disabled={datasetsLoading}
            >
              {datasetsLoading ? "Refreshing..." : "Refresh Datasets"}
            </button>

            <button
              type="submit"
              disabled={loading || !dataset || datasetsLoading || (compareMode && selectedStrategies.length === 0)}
            >
              {loading ? "Running..." : compareMode ? "Run Comparison" : "Run Backtest"}
            </button>
          </div>
        </form>

        <form className="upload-panel" onSubmit={handleUpload}>
          <label>
            Upload CSV
            <input name="datasetFile" type="file" accept=".csv,text/csv" />
          </label>
          <button type="submit" disabled={uploading}>
            {uploading ? "Uploading..." : "Add Dataset"}
          </button>
        </form>
      </section>

      {error ? <p className="error-banner">{error}</p> : null}

      {result ? (
        <section className="results-grid">
          {mlMetadataList.length > 0 ? (
            <article className="info-card">
              <h2>ML Snapshot</h2>
              <div className="info-grid">
                {mlMetadataList.map((item) => (
                  <div key={`${item.model_name}-${item.validation_mode}`} className="info-block">
                    <strong>{formatStrategy(item.model_name)}</strong>
                    <p>Accuracy: {item.accuracy.toFixed(4)}</p>
                    <p>Validation: {item.validation_mode}</p>
                    <p>Predictions: {item.num_predictions}</p>
                  </div>
                ))}
              </div>
            </article>
          ) : null}

          <article className="comparison-card">
            <div className="comparison-header">
              <div>
                <p className="eyebrow">{result.asset}</p>
                <h2>Strategy Comparison</h2>
              </div>
              <span className="trade-pill">{result.results.length} strategies</span>
            </div>

            <MetricsTable results={result.results} />
          </article>

          <MultiSeriesChart
            title="Equity Curve Comparison"
            timeline={timeline}
            series={chartSeries}
            formatValue={formatCurrency}
          />

          <MultiSeriesChart
            title="Drawdown Comparison"
            timeline={timeline}
            series={drawdownSeries}
            formatValue={formatPercent}
          />

          <section className="results-card-grid">
            {result.results.map((item, index) => (
              <article className="result-card" key={item.strategy_name}>
                <div className="result-header">
                  <div>
                    <p className="eyebrow">{item.asset_name}</p>
                    <h2>{formatStrategy(item.strategy_name)}</h2>
                  </div>
                  <span
                    className="trade-pill"
                    style={{
                      backgroundColor: `${SERIES_COLORS[index % SERIES_COLORS.length]}22`,
                      color: SERIES_COLORS[index % SERIES_COLORS.length],
                    }}
                  >
                    {item.trades.length} trades
                  </span>
                </div>

                <div className="metric-grid">
                  <MetricCard label="Total Return" value={formatPercent(item.metrics.total_return)} />
                  <MetricCard label="Sharpe Ratio" value={formatNumber(item.metrics.sharpe_ratio)} />
                  <MetricCard label="Max Drawdown" value={formatPercent(item.metrics.max_drawdown)} />
                  <MetricCard label="Final Equity" value={formatCurrency(item.metrics.final_equity)} />
                </div>
              </article>
            ))}
          </section>
        </section>
      ) : (
        <section className="empty-state">
          <p>Run a backtest or comparison to see metrics, overlays, and drawdown behavior here.</p>
        </section>
      )}
    </main>
  );
}

function MetricsTable({ results }: { results: BacktestResult[] }) {
  return (
    <div className="table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Total Return</th>
            <th>Sharpe</th>
            <th>Max Drawdown</th>
            <th>Final Equity</th>
            <th>Trades</th>
            <th>Exposure</th>
          </tr>
        </thead>
        <tbody>
          {results.map((item) => (
            <tr key={item.strategy_name}>
              <td>{formatStrategy(item.strategy_name)}</td>
              <td>{formatPercent(item.metrics.total_return)}</td>
              <td>{formatNumber(item.metrics.sharpe_ratio)}</td>
              <td>{formatPercent(item.metrics.max_drawdown)}</td>
              <td>{formatCurrency(item.metrics.final_equity)}</td>
              <td>{String(item.metrics.num_trades ?? item.trades.length)}</td>
              <td>{formatPercent(item.metrics.exposure_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MultiSeriesChart({
  title,
  timeline,
  series,
  formatValue,
}: {
  title: string;
  timeline: string[];
  series: { label: string; values: number[]; color: string }[];
  formatValue: (value: unknown) => string;
}) {
  const width = 960;
  const height = 260;
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  if (series.length === 0) {
    return null;
  }

  const allValues = series.flatMap((item) => item.values);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const range = max - min || 1;

  const normalizedSeries = series.map((item) => ({
    ...item,
    points: item.values
      .map((value, index) => {
        const x = (index / Math.max(item.values.length - 1, 1)) * width;
        const y = height - ((value - min) / range) * height;
        return `${x},${y}`;
      })
      .join(" "),
  }));

  const activeIndex = hoverIndex ?? series[0].values.length - 1;
  const activeDate = timeline[activeIndex] ?? "-";

  return (
    <article className="chart-panel">
      <div className="chart-panel-header">
        <div>
          <h2>{title}</h2>
          <p>{activeDate}</p>
        </div>
        <div className="legend-row">
          {series.map((item) => (
            <span key={item.label} className="legend-chip">
              <i style={{ backgroundColor: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      </div>

      <div className="chart-tooltip-grid">
        {series.map((item) => (
          <div key={item.label} className="chart-tooltip-card">
            <span>{item.label}</span>
            <strong>{formatValue(item.values[activeIndex])}</strong>
          </div>
        ))}
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="chart-svg comparison-svg"
        preserveAspectRatio="none"
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / rect.width;
          const nextIndex = Math.min(
            Math.max(Math.round(ratio * Math.max(timeline.length - 1, 0)), 0),
            Math.max(timeline.length - 1, 0),
          );
          setHoverIndex(nextIndex);
        }}
        onMouseLeave={() => setHoverIndex(null)}
      >
        {normalizedSeries.map((item) => (
          <polyline
            key={item.label}
            fill="none"
            stroke={item.color}
            strokeWidth="3"
            strokeLinejoin="round"
            strokeLinecap="round"
            points={item.points}
          />
        ))}
        {hoverIndex !== null ? (
          <line
            x1={(hoverIndex / Math.max(timeline.length - 1, 1)) * width}
            x2={(hoverIndex / Math.max(timeline.length - 1, 1)) * width}
            y1={0}
            y2={height}
            stroke="rgba(31, 41, 51, 0.25)"
            strokeDasharray="4 4"
          />
        ) : null}
      </svg>
    </article>
  );
}

function toggleStrategy(current: string[], value: string) {
  return current.includes(value) ? current.filter((item) => item !== value) : [...current, value];
}

function normalizeMlMetadata(value: ApiResponse["ml_metadata"]) {
  if (!value) {
    return [];
  }
  return Array.isArray(value) ? value : [value];
}

function formatStrategy(value: string) {
  return value.replaceAll("_", " ");
}

function formatNumber(value: unknown) {
  if (typeof value !== "number") {
    return "-";
  }
  return value.toFixed(3);
}

function formatPercent(value: unknown) {
  if (typeof value !== "number") {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatCurrency(value: unknown) {
  if (typeof value !== "number") {
    return "-";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}
