ARG DB_USER
ARG DB_PASS
ARG DB_HOST
ARG DB_PORT
ARG DB_NAME
ARG PATH_TO_CERT
ARG QDRANT_URL
ARG OPENROUTER_API_KEY

FROM python:3.13.5-slim

ENV DB_USER=$DB_USER
ENV DB_PASS=$DB_PASS
ENV DB_HOST=$DB_HOST
ENV DB_PORT=$DB_PORT
ENV DB_NAME=$DB_NAME
ENV PATH_TO_CERT=$PATH_TO_CERT
ENV QDRANT_URL=$QDRANT_URL
ENV OPENROUTER_API_KEY=$OPENROUTER_API_KEY

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/
COPY frontend.py ./
COPY .streamlit/ ./.streamlit/
COPY root.crt ./

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "frontend.py"]