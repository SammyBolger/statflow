# Reproducible container for the StatFlow dashboard and MLflow UI.
#
# One image, two services in docker-compose. The image ships the code + all
# runtime deps; data and MLflow state come from bind-mounted host directories
# so the container stays stateless.

FROM python:3.11-slim

# Copy the uv static binary in — avoids pip installing uv into the image.
COPY --from=ghcr.io/astral-sh/uv:0.5.7 /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first so the layer cache survives code edits.
COPY pyproject.toml uv.lock .python-version ./

# Install runtime deps only (no dev extras).
# --frozen locks to uv.lock exactly — no lockfile drift in production images.
RUN uv sync --frozen --no-dev

# Now copy the source tree.
COPY src ./src

# Streamlit default port.
EXPOSE 8501

# docker-compose overrides this with the actual service command.
CMD ["uv", "run", "streamlit", "run", "src/statflow/dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
