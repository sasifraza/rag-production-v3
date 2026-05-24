#!/usr/bin/env bash
set -euo pipefail

docker buildx build --platform linux/amd64 \
  -t ragv3acr.azurecr.io/rag-production-v3:latest \
  .

az acr update --admin-enabled true --name ragv3acr

docker login ragv3acr.azurecr.io \
  --username $(az acr credential show --name ragv3acr --query username -o tsv) \
  --password $(az acr credential show --name ragv3acr --query passwords[0].value -o tsv)

docker push ragv3acr.azurecr.io/rag-production-v3:latest
