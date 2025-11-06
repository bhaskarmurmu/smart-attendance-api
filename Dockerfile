FROM python:3.10-slim

# Install system build deps needed for dlib / face-recognition and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libx11-6 \
    libgtk-3-dev \
    python3-dev \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install: upgrade pip/setuptools/wheel first
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

ENV PORT=8000

EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
