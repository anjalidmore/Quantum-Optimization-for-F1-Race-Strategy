# F1 Race Strategy Intelligence — dashboard (Next.js 15 / React 19)
#
# Node 20 LTS: frontend/package.json declares no "engines" field, and the
# root README's stated requirement is "Node 20+".
#
# NEXT_PUBLIC_* values are inlined into the client bundle at `next build`
# time (the same reason ./run.sh deletes a stale frontend/.next before
# `next dev` — see run.sh's step 5 comment). So NEXT_PUBLIC_API_BASE_URL is
# taken as a build arg here, not just a runtime env var: the value baked in
# at build time is what actually reaches the browser. docker-compose.yml
# passes it as both a build arg (to bake it in) and a runtime env (for any
# server-side code path that reads it after start).

# ---------------------------------------------------------------------------
# deps: install once, reused by the build stage. Kept separate so an
# application-code-only change doesn't reinstall node_modules.
# ---------------------------------------------------------------------------
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---------------------------------------------------------------------------
# builder: produce the production build.
# ---------------------------------------------------------------------------
FROM node:20-slim AS builder
WORKDIR /app
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# This project has no frontend/public directory today; create an empty one
# so the runner stage's COPY below has something to copy regardless (Next
# serves an empty public/ fine, and this keeps the Dockerfile correct if one
# is added later).
RUN mkdir -p public
RUN npm run build

# ---------------------------------------------------------------------------
# runner: slim runtime image — only what `next start` needs.
# ---------------------------------------------------------------------------
FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/package.json ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules

EXPOSE 3000

CMD ["npm", "run", "start", "--", "-p", "3000", "-H", "0.0.0.0"]
