#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOMAIN="${DOMAIN:-app.example.com}"
CERT_FULLCHAIN="${CERT_FULLCHAIN:-}"
CERT_PRIVKEY="${CERT_PRIVKEY:-}"

if [[ -z "${CERT_FULLCHAIN}" || -z "${CERT_PRIVKEY}" ]]; then
  echo "ERROR: CERT_FULLCHAIN and CERT_PRIVKEY must be set."
  echo "Example:"
  echo "  DOMAIN=app.yourdomain.com CERT_FULLCHAIN=/etc/letsencrypt/live/app.yourdomain.com/fullchain.pem CERT_PRIVKEY=/etc/letsencrypt/live/app.yourdomain.com/privkey.pem ./deploy/scripts/deploy.sh"
  exit 1
fi

mkdir -p "${PROJECT_DIR}/deploy/nginx/certs"
cp "${CERT_FULLCHAIN}" "${PROJECT_DIR}/deploy/nginx/certs/fullchain.pem"
cp "${CERT_PRIVKEY}" "${PROJECT_DIR}/deploy/nginx/certs/privkey.pem"

tmp_conf="$(mktemp)"
sed "s/__DOMAIN__/${DOMAIN}/g" "${PROJECT_DIR}/deploy/nginx/nginx.conf" > "${tmp_conf}"
mv "${tmp_conf}" "${PROJECT_DIR}/deploy/nginx/default.conf"

cd "${PROJECT_DIR}"
docker compose up -d --build
docker compose ps

echo "Deployment completed."
echo "URL: https://${DOMAIN}"
