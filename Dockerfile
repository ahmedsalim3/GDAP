FROM python:3.12-slim

WORKDIR /app

COPY configs/requirements.txt .

RUN pip install -U pip

RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

EXPOSE 8501  

ENV PYTHONPATH="/app:/src:/mount/src:/mount/app"

CMD ["streamlit", "run", "app/Home.py"]