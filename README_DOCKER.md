# Docker deployment (Railway / local)

This project uses native extensions (dlib / face-recognition) which require system build tools. Use the included Dockerfile to build and deploy the app (recommended for Railway).

Local build and run (macOS, Docker installed):

```bash
# build image
docker build -t face-recog-api:latest .

# run container (maps port 8000)
docker run --rm -p 8000:8000 -e PORT=8000 face-recog-api:latest
```

Then open http://localhost:8000 for the root endpoint.

Deploying to Railway

- In Railway, create a new project and select GitHub repo (this repo), then choose "Dockerfile" as the deployment method. Railway will build the Dockerfile and run the container.
- Make sure the `PORT` environment variable is set by Railway (Railway usually provides it automatically). The Dockerfile uses `ENV PORT=8000` and exposes 8000.

Notes and troubleshooting

- If the build fails due to missing libraries, ensure the Dockerfile's apt packages are sufficient for the target platform. You can add additional apt packages if errors suggest missing libs.
- For faster iteration during development, you can install dlib locally in a dev machine that has build tools and then use a multi-stage or prebuilt wheel approach. For deployment, Docker is the most reliable.
