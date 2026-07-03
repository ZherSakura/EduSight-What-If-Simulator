# EduSight backend
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/

COPY ["Data_ML_Engine/risk_engine.py", "/app/Data_ML_Engine/risk_engine.py"]
COPY ["Data_ML_Engine/recommendation_engine.py", "/app/Data_ML_Engine/recommendation_engine.py"]

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
