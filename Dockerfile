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
COPY muaddib/ ./muaddib/

# Install Python dependencies
RUN uv sync --frozen

RUN mkdir -p artifacts/ /data /home/irssi/.irssi

# Default command
CMD ["uv", "run", "python", "-m", "muaddib.main"]
