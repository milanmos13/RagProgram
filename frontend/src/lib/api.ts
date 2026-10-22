export type Origin = "prose" | "infobox" | "table";

export type SearchResult = {
  id: string;
  article_title: string;
  section_path: string[];
  origin: Origin;
  text: string;
  score: number;
  revision_url: string;
};

export type SearchResponse = {
  configuration: string;
  results: SearchResult[];
};

// Set at build time (Vercel project settings); inlined into the client bundle.
const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

export async function search(question: string): Promise<SearchResponse> {
  const response = await fetch(`${API_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    throw new Error(`Search failed with status ${response.status}`);
  }
  return response.json();
}

export type Metrics = {
  recall_at_1: number;
  recall_at_5: number;
  recall_at_10: number;
  mrr_at_10: number;
};

export type EvalResults = {
  created_at: string;
  configuration: string;
  split: "dev" | "test" | "all";
  manifest_version: string;
  question_count: number;
  metrics: Metrics;
  multi_hop: {
    question_count: number;
    all_evidence: Metrics;
    any_evidence: Metrics;
  };
};

/** The latest eval run, or null when none has been run yet. */
export async function getResults(): Promise<EvalResults | null> {
  const response = await fetch(`${API_URL}/results`);
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`Loading results failed with status ${response.status}`);
  }
  return response.json();
}
