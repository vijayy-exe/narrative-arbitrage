// API client for the Phase 7 FastAPI backend (backend/main.py). Base URL is
// configurable via NEXT_PUBLIC_API_BASE_URL (see frontend/.env.local.example)
// -- defaults to localhost:8000, where `uvicorn backend.main:app` runs.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export type Mode = "live" | "replay";
export type Method = "ppo" | "momentum_baseline";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(`${res.status} ${path}: ${detail}`);
  }
  return res.json();
}

export interface AssetInfo {
  symbol: string;
  category: "stock" | "crypto";
  method: Method;
}

export async function listAssets(): Promise<{ assets: AssetInfo[] }> {
  return getJSON(`/api/assets`);
}

export interface FeatureHistoryPoint {
  date: string;
  features: Record<string, number | null>;
}

export interface FeaturesResponse {
  mode: Mode;
  asset: string;
  date: string;
  features: Record<string, number | null>;
  history: FeatureHistoryPoint[];
}

export async function getFeatures(
  asset: string,
  mode: Mode = "live"
): Promise<FeaturesResponse> {
  return getJSON(`/api/features/${asset}?mode=${mode}`);
}

export interface Decision {
  asset: string;
  method: Method;
  raw_action?: number | null;
  target_position: number;
  price: number;
  reasoning: string;
  timestamp?: string;
  date?: string;
}

export interface DecisionResponse extends Decision {
  mode: Mode;
  history: Decision[];
}

export async function getDecision(
  asset: string,
  mode: Mode = "live"
): Promise<DecisionResponse> {
  return getJSON(`/api/decision/${asset}?mode=${mode}`);
}

export interface PortfolioAssetRow {
  method: Method;
  cash: number;
  position: number;
  entry_price: number | null;
  last_price: number | null;
  unrealized_pnl: number;
  last_updated?: string;
}

export interface PortfolioResponse {
  mode: Mode;
  assets: Record<string, PortfolioAssetRow>;
  total_portfolio_value: number;
}

export async function getPortfolio(
  mode: Mode = "live"
): Promise<PortfolioResponse> {
  return getJSON(`/api/portfolio?mode=${mode}`);
}

export interface BacktestResponse {
  asset: string;
  methods: Record<string, Record<string, number>>;
}

export async function getBacktest(asset: string): Promise<BacktestResponse> {
  return getJSON(`/api/backtest/${asset}`);
}
