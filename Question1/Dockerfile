FROM python:3.10-slim
WORKDIR /app
RUN pip install transformers sacrebleu sentencepiece torch --quiet
COPY translate.py .
COPY input.txt .
COPY reference.txt .
CMD ["python", "translate.py"]
