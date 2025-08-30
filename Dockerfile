FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        curl \
        git \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY irssi_llmagent/ ./irssi_llmagent/

# Install Python dependencies
RUN uv sync --frozen

# Create directory for varlink socket
RUN mkdir -p /home/irssi/.irssi

# Default command
CMD ["uv", "run", "python", "-m", "irssi_llmagent.main"]
