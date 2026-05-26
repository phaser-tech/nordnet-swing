#!/usr/bin/env bash
# Verifies that the development environment is correctly set up.
# Run after `uv sync` and `docker compose up -d`.

set -euo pipefail

echo "=== Nordnet Swing setup verification ==="
echo

# Check Python version
echo "→ Python version:"
uv run python --version
echo

# Check key packages installed
echo "→ Key packages:"
uv run python -c "import pydantic; print(f'  pydantic    {pydantic.__version__}')"
uv run python -c "import pandas; print(f'  pandas      {pandas.__version__}')"
uv run python -c "import numpy; print(f'  numpy       {numpy.__version__}')"
uv run python -c "import yfinance; print(f'  yfinance    {yfinance.__version__}')"
echo

# Check Docker container is up
echo "→ Docker:"
if docker ps --filter "name=nordnet-swing-postgres" --filter "status=running" | grep -q nordnet-swing-postgres; then
    echo "  ✓ PostgreSQL container running"
else
    echo "  ✗ PostgreSQL container NOT running"
    echo "    Run: docker compose up -d"
    exit 1
fi
echo

# Check Postgres is responding
echo "→ PostgreSQL connection:"
if docker exec nordnet-swing-postgres pg_isready -U nordnet_swing > /dev/null 2>&1; then
    echo "  ✓ PostgreSQL accepting connections"
else
    echo "  ✗ PostgreSQL not responding"
    exit 1
fi
echo

# Run tests
echo "→ Running tests:"
uv run pytest -q
echo

# Run type check
echo "→ Type check:"
uv run mypy packages/ --no-error-summary && echo "  ✓ mypy clean" || echo "  ✗ mypy errors"
echo

# Run linter
echo "→ Linter:"
uv run ruff check . --quiet && echo "  ✓ ruff clean" || echo "  ✗ ruff issues"
echo

echo "=== Setup verified. Ready to build. ==="
