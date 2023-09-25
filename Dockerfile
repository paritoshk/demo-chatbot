FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y poppler-utils && apt-get install -y tesseract-ocr

COPY app.py .
COPY file.py .

EXPOSE 8503

CMD ["streamlit", "run", "app.py", "--server.port", "8503"]
