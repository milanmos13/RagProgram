# Deploying

The backend runs in a Hugging Face Docker Space and the frontend on Vercel. The Space
builds the last stage of `backend/Dockerfile` (`prod`). That stage fetches the
manifest's pinned Articles, then chunks and embeds them, and bakes the result and the
model into the image. Nothing built lives in git.

## 1. Backend: Hugging Face Space

1. Create a Space at https://huggingface.co/new-space. Choose the **Docker** SDK with a
   blank template and CPU basic hardware, e.g. `<hf-user>/nba-rag`.
2. In the Space's **Settings → Variables and secrets**, add these variables:
   - `NBA_RAG_CONTACT` (recommended): your email or URL for the Wikimedia user agent.
     The Space passes it to the build as a build arg. If unset, the build uses the
     project's GitHub issues URL.
   - `NBA_RAG_ALLOWED_ORIGINS`: the Vercel production URL, e.g.
     `https://nba-rag.vercel.app`. Separate several origins with commas. Until you
     know the URL, use `http://localhost:3000`.
3. Upload the backend directory. The Space repo root is `backend/`, and its
   `README.md` holds the Space config (`sdk: docker`, `app_port: 7860`):

   ```bash
   hf auth login
   hf upload <hf-user>/nba-rag backend . --repo-type space \
     --exclude "data/*" "tests/*" ".venv/*" "**/__pycache__/*"
   ```

4. Wait for the build (a few minutes: torch, the Wikipedia fetch and embedding). The
   API is then live at `https://<hf-user>-nba-rag.hf.space`. Check it:

   ```bash
   curl -X POST https://<hf-user>-nba-rag.hf.space/search \
     -H 'content-type: application/json' -d '{"question": "Who won the 2016 NBA Finals?"}'
   ```

To redeploy after a backend change, repeat step 3.

## 2. Frontend: Vercel

1. Import `milanmos13/RagProgram` at https://vercel.com/new and set **Root Directory**
   to `frontend`. The framework preset is detected as Next.js.
2. Add the environment variable `NEXT_PUBLIC_API_URL=https://<hf-user>-nba-rag.hf.space`.
   It is inlined at build time, so you must redeploy after changing it.
3. Deploy. Pushes to `main` redeploy automatically.

## 3. Connect them

Set the Space variable `NBA_RAG_ALLOWED_ORIGINS` to the Vercel production URL. Saving
a variable restarts the Space; if it doesn't, use **Restart** in the Space menu (no
rebuild needed, since this is a runtime variable). Open the Vercel URL and run a search.

The API rejects browser requests from any other origin, including Vercel preview
deployments.

## Checking the production image locally

```bash
docker build --target prod -t nba-rag-backend:prod backend
scripts/smoke-test-image.sh nba-rag-backend:prod   # serves a search offline as UID 1000
```

CI runs the same two commands on every push.
