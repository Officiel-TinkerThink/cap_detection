# ============================
# Builder Image
# ============================
FROM python:3.12-slim AS builder
WORKDIR /app

# Install only what is absolutely required
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install deps into a wheelhouse (NO system install yet)
RUN uv pip wheel . -w /tmp/wheels

# Copy project files
COPY bsort ./bsort
COPY configs ./configs
COPY tools ./tools
COPY scripts ./scripts


# ============================
# Final Runtime Image
# ============================
FROM python:3.12-slim
WORKDIR /app

# Install only runtime libs needed by your app (small!)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels built earlier
COPY --from=builder /tmp/wheels /tmp/wheels

# Install deps from wheelhouse (fast + tiny footprint)
RUN pip install --no-cache-dir /tmp/wheels/*

# Copy project files
COPY bsort ./bsort
COPY configs ./configs
COPY tools ./tools
COPY scripts ./scripts

ENTRYPOINT ["bsort"]
