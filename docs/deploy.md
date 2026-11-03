# Deploying

The backend runs on Google Cloud Run and the frontend on Vercel.

`gcloud builds submit` uploads `backend/` to Cloud Build, which builds the last stage
of `backend/Dockerfile` (`prod`): it fetches the manifest's pinned Articles, chunks and
embeds them, and bakes the result and the model into the image. `gcloud run deploy`
then serves that image. Nothing built lives in git. The service scales to zero when
idle, so the first request after a quiet spell waits 10–20 seconds for the container to
start.

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

The build fetches 266 Articles politely (one request a second) and embeds about 11k
Chunks. Don't use `gcloud run deploy --source`: it builds on Cloud Build's default
machine, where the Corpus build runs past the 30-minute limit, and it can't pick a
bigger machine without a private worker pool. Build the image yourself instead, then
deploy it.

Create the image repository once:

```bash
gcloud artifacts repositories create cloud-run-source-deploy \
  --repository-format docker --location us-central1
```

Then, from the repo root, build the image on an 8-CPU machine:

```bash
IMAGE=us-central1-docker.pkg.dev/$(gcloud config get project)/cloud-run-source-deploy/nba-rag:$(git rev-parse --short HEAD)
gcloud builds submit backend --tag "$IMAGE" --machine-type e2-highcpu-8 --timeout 3600s
```

- The build takes about 25–30 minutes: torch, the Wikipedia fetch and embedding. The
  embedding step prints nothing until it ends with
  `Built 11373 Chunks from 266 Articles`.
- `e2-highcpu-8` costs more per build-minute than the default machine. See
  https://cloud.google.com/build/pricing.

Deploy the image:

```bash
gcloud run deploy nba-rag --image "$IMAGE" \
  --memory 2Gi --cpu 1 \
  --min-instances 0 --max-instances 2 \
  --allow-unauthenticated \
  --set-env-vars NBA_RAG_ALLOWED_ORIGINS=http://localhost:3000
```

It prints the service URL. Check it:

```bash
curl -X POST <service-url>/search \
  -H 'content-type: application/json' -d '{"question": "Who won the 1947 BAA Finals?"}'
```

The top result should be the `1947 BAA Finals` Article, and `GET <service-url>/results`
should report the manifest version in `backend/manifest.json`.

Notes:
- `--max-instances 2` caps how far a traffic spike can scale, and so caps the cost.
- `--allow-unauthenticated` makes the API public. Browsers still need an allowed origin
  (step 4).
- The Wikimedia user agent's contact defaults to this project's GitHub issues URL.
  `gcloud builds submit --tag` can't pass build args, so to use your own address, change
  the `NBA_RAG_CONTACT` default in `backend/Dockerfile`.

To redeploy after a backend change, build a new image as above, then deploy it
**without** `--set-env-vars`, which would reset the allowed origins. Redeploys keep the
variables and limits already set on the service:

```bash
gcloud run deploy nba-rag --image "$IMAGE"
```

## 3. Frontend: Vercel

1. Import `milanmos13/RagProgram` at https://vercel.com/new and set **Root Directory**
   to `frontend`. The framework preset is detected as Next.js.
2. Add the environment variable
   `NEXT_PUBLIC_API_URL=<service-url>`. It is inlined at build
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
