# 1) Lightweight Python base image
FROM python:3.12-slim

# 2) Set working directory inside the container
WORKDIR /app

# 3) Copy dependencies separately (Docker layer caching)
COPY requirements.txt .

# 4) Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy the remaining application code
COPY . .

# 6) Document the exposed port
EXPOSE 8000

# 7) Start the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]