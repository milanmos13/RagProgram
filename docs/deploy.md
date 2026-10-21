# Deploying

The backend runs on Google Cloud Run and the frontend on Vercel.

`gcloud run deploy --source backend` uploads `backend/` to Cloud Build. Cloud Build
builds the last stage of `backend/Dockerfile` (`prod`), which fetches the manifest's
pinned Articles, chunks and embeds them, and bakes the result and the model into the
image. Nothing built lives in git. The service scales to zero when idle, so the first
request after a quiet spell waits 10–20 seconds for the container to start.

## 1. One-time Google Cloud setup

1. Create a project at https://console.cloud.google.com/projectcreate, e.g. `nba-rag`,
   and link a billing account. Light demo traffic stays within the Cloud Run free tier,
   but a billing account is still required. Stored images may cost a few cents a month.
   See https://cloud.google.com/run/pricing.
2. Install the CLI and log in:

   ```bash
   brew install --cask google-cloud-sdk
   gcloud auth login
   gcloud config set project <project-id>
   gcloud config set run/region us-central1
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

## 2. Backend: Cloud Run

Deploy from the repo root:

```bash
gcloud run deploy nba-rag --source backend \
  --port 7860 --memory 2Gi --cpu 1 \
  --min-instances 0 --max-instances 2 \
  --allow-unauthenticated \
  --set-env-vars NBA_RAG_ALLOWED_ORIGINS=http://localhost:3000
```

- The first deploy asks to create an Artifact Registry repository
  (`cloud-run-source-deploy`). Answer yes.
- The build takes about 5–10 minutes: torch, the Wikipedia fetch and embedding. The
  build log should show `Built 466 Chunks from 5 Articles`.
- When it finishes, it prints the service URL, e.g.
  `https://nba-rag-<hash>-uc.a.run.app`. Check it:

  ```bash
  curl -X POST https://nba-rag-<hash>-uc.a.run.app/search \
    -H 'content-type: application/json' -d '{"question": "Who won the 2016 NBA Finals?"}'
  ```

Notes:
- `--max-instances 2` caps how far a traffic spike can scale, and so caps the cost.
- `--allow-unauthenticated` makes the API public. Browsers still need an allowed origin
  (step 4).
- The Wikimedia user agent's contact defaults to this project's GitHub issues URL.
  Source deploys can't pass build args, so to use your own address, change the
  `NBA_RAG_CONTACT` default in `backend/Dockerfile`.

To redeploy after a backend change, run the command again **without**
`--set-env-vars`, which would reset the allowed origins. Redeploys keep the variables
already set on the service:

```bash
gcloud run deploy nba-rag --source backend
```

## 3. Frontend: Vercel

1. Import `milanmos13/RagProgram` at https://vercel.com/new and set **Root Directory**
   to `frontend`. The framework preset is detected as Next.js.
2. Add the environment variable
   `NEXT_PUBLIC_API_URL=https://nba-rag-<hash>-uc.a.run.app`. It is inlined at build
   time, so you must redeploy after changing it.
3. Deploy. Pushes to `main` redeploy automatically.

## 4. Connect them

Allow the Vercel production URL to call the API. This creates a new revision with the
updated variable; it doesn't rebuild the image:

```bash
gcloud run services update nba-rag \
  --update-env-vars NBA_RAG_ALLOWED_ORIGINS=https://<your-app>.vercel.app
```

To allow several origins, change gcloud's list delimiter so the commas survive:

```bash
gcloud run services update nba-rag \
  --update-env-vars '^@^NBA_RAG_ALLOWED_ORIGINS=https://<your-app>.vercel.app,http://localhost:3000'
```

Open the Vercel URL and run a search. The API rejects browser requests from any other
origin, including Vercel preview deployments.

## Checking the production image locally

```bash
docker build --target prod -t nba-rag-backend:prod backend
scripts/smoke-test-image.sh nba-rag-backend:prod   # serves a search offline as UID 1000
```

CI runs the same two commands on every push.
