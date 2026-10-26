"use client";

import { useEffect, useState } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { type EvalResults, getResults, type Metrics } from "@/lib/api";

type State =
  | { kind: "loading" }
  | { kind: "error" }
  | { kind: "empty" }
  | { kind: "done"; results: EvalResults };

export function Results() {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    getResults()
      .then((results) => {
        if (!cancelled) setState(results ? { kind: "done", results } : { kind: "empty" });
      })
      .catch(() => {
        if (!cancelled) setState({ kind: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  switch (state.kind) {
    case "loading":
      return (
        <div className="flex flex-col gap-3">
          <p role="status" className="sr-only">
            Loading results…
          </p>
          <Skeleton className="h-5 w-64" />
          <Skeleton className="h-40 w-full" />
        </div>
      );
    case "error":
      return (
        <p role="alert" className="text-sm text-destructive">
          Couldn&apos;t load results. Please try again in a moment.
        </p>
      );
    case "empty":
      return <p className="text-sm text-muted-foreground">No eval results yet.</p>;
    case "done":
      return <ResultsTable results={state.results} />;
  }
}

function ResultsTable({ results }: { results: EvalResults }) {
  const { multi_hop: multiHop } = results;
  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-muted-foreground">
        Configuration <span className="font-medium text-foreground">{results.configuration}</span>{" "}
        · split: {results.split} · manifest: {results.manifest_version}
      </p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Scope</TableHead>
            <TableHead className="text-right">Questions</TableHead>
            <TableHead className="text-right">Recall@1</TableHead>
            <TableHead className="text-right">Recall@5</TableHead>
            <TableHead className="text-right">Recall@10</TableHead>
            <TableHead className="text-right">MRR@10</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <MetricsRow label="All questions" count={results.question_count} metrics={results.metrics} />
          {multiHop.question_count > 0 && (
            <>
              <MetricsRow
                label="Multi-hop, all Evidence"
                count={multiHop.question_count}
                metrics={multiHop.all_evidence}
              />
              <MetricsRow
                label="Multi-hop, any Evidence"
                count={multiHop.question_count}
                metrics={multiHop.any_evidence}
              />
            </>
          )}
        </TableBody>
      </Table>
      <p className="text-sm text-muted-foreground">
        A question counts as found at the rank by which all of its Evidence has been retrieved.
        For Multi-hop questions, any Evidence instead uses the rank of the first Chunk containing
        any one span.
      </p>
    </div>
  );
}

function MetricsRow({ label, count, metrics }: { label: string; count: number; metrics: Metrics }) {
  const values = [metrics.recall_at_1, metrics.recall_at_5, metrics.recall_at_10, metrics.mrr_at_10];
  return (
    <TableRow>
      <TableCell className="font-medium">{label}</TableCell>
      <TableCell className="text-right tabular-nums">{count}</TableCell>
      {values.map((value, i) => (
        <TableCell key={i} className="text-right font-mono tabular-nums">
          {value.toFixed(3)}
        </TableCell>
      ))}
    </TableRow>
  );
}
