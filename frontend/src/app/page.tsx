import { Search } from "@/components/search";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-4 py-12 sm:py-20">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight">NBA History RAG</h1>
        <p className="text-muted-foreground">
          Ask a question about NBA history and see the Wikipedia Chunks the retriever ranks
          highest.
        </p>
      </header>
      <Search />
    </main>
  );
}
