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
