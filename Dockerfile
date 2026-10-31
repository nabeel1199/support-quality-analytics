FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so this layer is cached until the requirements change.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

RUN useradd --create-home app && chown -R app /app
USER app

# Default: generate the synthetic data and run the whole SQL pipeline.
# Run the tests instead with: docker run --rm support-quality-analytics pytest
CMD ["sh", "-c", "python scripts/generate_data.py && python src/pipeline.py"]
