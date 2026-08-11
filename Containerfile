FROM python:3.12-slim

# Create a non-root user for security
RUN useradd -m -s /bin/bash appuser

WORKDIR /app

# Install project and dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache -e .

# Copy the rest of the application
COPY . .

# Ensure correct permissions
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# MLflow tracking URI
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8501

CMD ["streamlit", "run", "src/crispdm/app/finapp.py", "--server.address", "0.0.0.0"]
