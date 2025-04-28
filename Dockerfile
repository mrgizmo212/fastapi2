FROM python:3.9-slim

WORKDIR /app

# Install dependencies with explicit pip upgrade and verbose logging
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --verbose -r requirements.txt && \
    pip install --no-cache-dir pytz

# Verify pytz is installed
RUN python -c "import pytz; print(f'pytz version: {pytz.__version__}')"

# Copy application code
COPY . .

# Set environment variables
ENV PORT=80

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4", "--timeout-keep-alive", "65"]
