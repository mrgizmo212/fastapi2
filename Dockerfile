FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Debug: List directory contents
RUN echo "Current directory:" && ls -la

# Copy application code
COPY . .

# Debug: List directory after copy
RUN echo "After copy:" && ls -la

# Set environment variables
ENV PORT=80

# Debug: Print Python version and installed packages
RUN python --version && pip list

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--log-level", "debug"]
