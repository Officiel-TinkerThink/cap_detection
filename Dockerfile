# ============================
# Builder Image
# ============================
FROM python:3.12-slim AS builder
WORKDIR /app

# Install system dependencies + uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

# Make uv available in PATH (uv installs into /root/.local/bin)
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./

# Copy project source (similar to original)
COPY bsort ./bsort
COPY configs ./configs
COPY tools ./tools
COPY scripts ./scripts

# Install dependencies + build wheels into /usr/local
RUN uv pip install --system --no-cache .


# ============================
# Final Runtime Image
# ============================
FROM python:3.12-slim
WORKDIR /app

# Copy installed site-packages + binaries from builder
COPY --from=builder /usr/local /usr/local

# Copy minimal project files only
COPY bsort ./bsort
COPY configs ./configs
COPY tools ./tools
COPY scripts ./scripts

# Entry point (same as original)
ENTRYPOINT ["bsort"]