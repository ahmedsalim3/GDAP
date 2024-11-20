FROM python:3.12-slim

WORKDIR /app

COPY src/ /app/src/
COPY setup.py .

RUN pip install -U pip
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py"]