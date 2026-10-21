import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { delay, http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { API_URL, server } from "@/test/server";
import { Search } from "./search";

const GAME_7_BLOCK = {
  id: "2016 NBA Finals#27",
  article_title: "2016 NBA Finals",
  section_path: ["Game summaries", "Game 7"],
  origin: "prose",
  text: 'LeBron James delivered what became known as "The Block" on a layup attempt by Andre Iguodala.',
  score: 0.683,
  revision_url: "https://en.wikipedia.org/w/index.php?title=2016_NBA_Finals&oldid=1381121302",
};

const CURRY_LEAD = {
  id: "Stephen Curry#0",
  article_title: "Stephen Curry",
  section_path: [],
  origin: "prose",
  text: "Wardell Stephen Curry II is an American professional basketball player.",
  score: 0.41,
  revision_url: "https://en.wikipedia.org/w/index.php?title=Stephen_Curry&oldid=1381099564",
};

async function ask(question: string, Component: typeof Search = Search) {
  const user = userEvent.setup();
  render(<Component />);
  await user.type(screen.getByRole("searchbox", { name: /question/i }), question);
  await user.click(screen.getByRole("button", { name: /search/i }));
}

describe("Search", () => {
  it("sends the question and renders ranked Chunks with their source details", async () => {
    let body: unknown;
    server.use(
      http.post(`${API_URL}/search`, async ({ request }) => {
        body = await request.json();
        return HttpResponse.json({ configuration: "baseline", results: [GAME_7_BLOCK, CURRY_LEAD] });
      }),
    );

    await ask("Who blocked Iguodala?");

    const results = await screen.findAllByRole("article");
    expect(body).toEqual({ question: "Who blocked Iguodala?" });
    expect(results).toHaveLength(2);

    const [first, second] = results;
    expect(first).toHaveTextContent("1");
    expect(first).toHaveTextContent("2016 NBA Finals");
    expect(first).toHaveTextContent("Game summaries › Game 7");
    expect(first).toHaveTextContent("prose");
    expect(first).toHaveTextContent("0.683");
    expect(first).toHaveTextContent('"The Block"');
    expect(screen.getAllByRole("link", { name: /view on wikipedia/i })[0]).toHaveAttribute(
      "href",
      GAME_7_BLOCK.revision_url,
    );

    expect(second).toHaveTextContent("Stephen Curry");
    expect(second).toHaveTextContent("Lead section");
    expect(second).toHaveTextContent("0.410");
  });

  it("shows a loading state while the search is in flight", async () => {
    server.use(
      http.post(`${API_URL}/search`, async () => {
        await delay(50);
        return HttpResponse.json({ configuration: "baseline", results: [CURRY_LEAD] });
      }),
    );

    await ask("Curry");

    expect(screen.getByRole("status")).toHaveTextContent(/searching/i);
    expect(screen.getByRole("button", { name: /search/i })).toBeDisabled();
    expect(await screen.findByRole("article")).toBeInTheDocument();
    expect(screen.queryByText(/searching/i)).not.toBeInTheDocument();
  });

  it("says so when no Chunks match", async () => {
    server.use(
      http.post(`${API_URL}/search`, () =>
        HttpResponse.json({ configuration: "baseline", results: [] }),
      ),
    );

    await ask("Cricket");

    expect(await screen.findByText(/no chunks found/i)).toBeInTheDocument();
    expect(screen.queryByRole("article")).not.toBeInTheDocument();
  });

  it("shows an error when the search fails", async () => {
    server.use(http.post(`${API_URL}/search`, () => new HttpResponse(null, { status: 500 })));

    await ask("Curry");

    expect(await screen.findByRole("alert")).toHaveTextContent(/search failed/i);
  });

  describe("backend URL", () => {
    afterEach(() => {
      vi.unstubAllEnvs();
    });

    it("comes from NEXT_PUBLIC_API_URL, even when set with a trailing slash", async () => {
      vi.stubEnv("NEXT_PUBLIC_API_URL", "https://someone-nba-rag.hf.space/");
      vi.resetModules();
      const { Search: DeployedSearch } = await import("./search");
      server.use(
        http.post("https://someone-nba-rag.hf.space/search", () =>
          HttpResponse.json({ configuration: "baseline", results: [CURRY_LEAD] }),
        ),
      );

      await ask("Curry", DeployedSearch);

      expect(await screen.findByRole("article")).toHaveTextContent("Stephen Curry");
    });
  });
});
