FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 

COPY src/ /app/src/
COPY app/ /app/app/
COPY docs/ /app/docs/

EXPOSE 8501

WORKDIR /app/app
CMD ["streamlit", "run", "app.py"]