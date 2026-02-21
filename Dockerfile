FROM python:3.9-slim
WORKDIR /app

# Install git (needed for HF credentials and Task 10)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*
# Run evaluation on your HF model when container starts
CMD ["python", "src/eval.py", "--model", "your-username/my-exam-model"]

CMD ["tail", "-f", "/dev/null"]