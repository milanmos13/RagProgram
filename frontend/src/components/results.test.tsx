import { render, screen } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { API_URL, server } from "@/test/server";
import { Results } from "./results";

const RESULTS = {
  created_at: "2026-10-21T08:00:00+00:00",
  configuration: "baseline",
  split: "all",
  manifest_version: "tracer-5",
  question_count: 10,
  metrics: { recall_at_1: 0.3, recall_at_5: 0.8, recall_at_10: 0.9, mrr_at_10: 0.4743 },
  multi_hop: {
    question_count: 2,
    all_evidence: { recall_at_1: 0, recall_at_5: 0.5, recall_at_10: 0.5, mrr_at_10: 0.1 },
    any_evidence: { recall_at_1: 0.5, recall_at_5: 1, recall_at_10: 1, mrr_at_10: 0.75 },
  },
  questions: [],
};

function serve(response: () => Response | Promise<Response>) {
  server.use(http.get(`${API_URL}/results`, response));
}

describe("Results", () => {
  it("renders the metrics table for the latest eval run", async () => {
    serve(() => HttpResponse.json(RESULTS));

    render(<Results />);

    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(screen.getByText(/baseline/)).toBeInTheDocument();
    expect(screen.getByText(/split: all/i)).toBeInTheDocument();
    expect(screen.getByText(/manifest: tracer-5/i)).toBeInTheDocument();

    const columns = screen.getAllByRole("columnheader").map((c) => c.textContent);
    expect(columns).toEqual(["Scope", "Questions", "Recall@1", "Recall@5", "Recall@10", "MRR@10"]);

    const [, overall, allEvidence, anyEvidence] = screen.getAllByRole("row");
    expect(overall).toHaveTextContent(/^All questions\s*10\s*0\.300\s*0\.800\s*0\.900\s*0\.474$/);
    expect(allEvidence).toHaveTextContent(
      /^Multi-hop, all Evidence\s*2\s*0\.000\s*0\.500\s*0\.500\s*0\.100$/,
    );
    expect(anyEvidence).toHaveTextContent(
      /^Multi-hop, any Evidence\s*2\s*0\.500\s*1\.000\s*1\.000\s*0\.750$/,
    );
  });

  it("explains how a question counts as found", async () => {
    serve(() => HttpResponse.json(RESULTS));

    render(<Results />);

    await screen.findByRole("table");
    expect(screen.getByText(/found at the rank by which all of its Evidence/i)).toBeInTheDocument();
    expect(screen.getByText(/any Evidence.*first Chunk containing any/i)).toBeInTheDocument();
  });

  it("omits Multi-hop rows when the run has no Multi-hop questions", async () => {
    serve(() =>
      HttpResponse.json({
        ...RESULTS,
        multi_hop: { ...RESULTS.multi_hop, question_count: 0 },
      }),
    );

    render(<Results />);

    await screen.findByRole("table");
    expect(screen.getAllByRole("row")).toHaveLength(2);
  });

  it("shows a loading state while results are fetched", async () => {
    serve(async () => {
      await delay(50);
      return HttpResponse.json(RESULTS);
    });

    render(<Results />);

    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("says so when no eval has been run yet", async () => {
    serve(() => HttpResponse.json({ detail: "No eval results yet" }, { status: 404 }));

    render(<Results />);

    expect(await screen.findByText(/no eval results yet/i)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("shows an error when results can't be loaded", async () => {
    serve(() => new HttpResponse(null, { status: 500 }));

    render(<Results />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/couldn't load results/i);
  });
});
