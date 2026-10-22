import type { Metadata } from "next";
import { Results } from "@/components/results";

export const metadata: Metadata = { title: "Results · NBA History RAG" };

export default function ResultsPage() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-4 py-12 sm:py-20">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">Results</h1>
        <p className="text-muted-foreground">
          How well the retriever finds the Evidence for each Eval question. A question counts
          as found at k when every Evidence span is in a top-k Chunk.
        </p>
      </header>
      <Results />
    </main>
  );
}
