FROM node:24-bookworm-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git ripgrep libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.5.2 \
    && npm cache clean --force

ENV UV_PROJECT_ENVIRONMENT=/opt/vmi-agent/.venv \
    VIRTUAL_ENV=/opt/vmi-agent/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/vmi-python \
    PATH=/opt/vmi-agent/.venv/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache
WORKDIR /opt/vmi-agent
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY vendor/pbasex-matlab/LICENSE.txt vendor/pbasex-matlab/LICENSE.txt
COPY src src
ARG VMI_PYTHON_VERSION=3.13
RUN uv sync --locked --python "${VMI_PYTHON_VERSION}" --no-dev --no-editable --no-cache \
    && python -c "import h5py, numpy, vmi_analysis"

COPY packaging/agent/entrypoint.sh /usr/local/bin/vmi-agent-entrypoint
STOPSIGNAL SIGTERM
ENTRYPOINT ["/bin/sh", "/usr/local/bin/vmi-agent-entrypoint"]
