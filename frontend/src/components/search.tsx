"use client";

import { type FormEvent, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { search, type SearchResult } from "@/lib/api";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error" }
  | { kind: "done"; results: SearchResult[] };

export function Search() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;
    setState({ kind: "loading" });
    try {
      const { results } = await search(trimmed);
      setState({ kind: "done", results });
    } catch {
      setState({ kind: "error" });
    }
  }

  const loading = state.kind === "loading";

  return (
    <div className="flex flex-col gap-6">
      <form onSubmit={onSubmit} role="search" className="flex gap-2">
        <Input
          type="search"
          aria-label="Question"
          placeholder="Who blocked Andre Iguodala in Game 7 of the 2016 Finals?"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          className="h-10"
        />
        <Button type="submit" disabled={loading} className="h-10 px-5">
          Search
        </Button>
      </form>

      {state.kind === "loading" && <Loading />}
      {state.kind === "error" && (
        <p role="alert" className="text-sm text-destructive">
          Search failed. Please try again in a moment.
        </p>
      )}
      {state.kind === "done" && state.results.length === 0 && (
        <p className="text-sm text-muted-foreground">No Chunks found for that question.</p>
      )}
      {state.kind === "done" && state.results.length > 0 && (
        <ol className="flex flex-col gap-3">
          {state.results.map((result, index) => (
            <li key={result.id}>
              <ResultCard result={result} rank={index + 1} />
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function Loading() {
  return (
    <div className="flex flex-col gap-3">
      <p role="status" className="sr-only">
        Searching…
      </p>
      {[0, 1, 2].map((i) => (
        <Skeleton key={i} className="h-28 w-full" />
      ))}
    </div>
  );
}

function ResultCard({ result, rank }: { result: SearchResult; rank: number }) {
  const section = result.section_path.length ? result.section_path.join(" › ") : "Lead section";
  return (
    <Card role="article" aria-labelledby={`result-${rank}`}>
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <CardTitle id={`result-${rank}`} className="flex items-baseline gap-2">
            <span className="text-muted-foreground tabular-nums">{rank}</span>
            {result.article_title}
          </CardTitle>
          <p className="text-sm text-muted-foreground">{section}</p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Badge variant="secondary">{result.origin}</Badge>
          <span className="font-mono text-sm tabular-nums" title="Score">
            {result.score.toFixed(3)}
          </span>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <p className="text-sm leading-6">{result.text}</p>
        <a
          href={result.revision_url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm font-medium text-primary underline-offset-4 hover:underline"
        >
          View on Wikipedia
        </a>
      </CardContent>
    </Card>
  );
}
