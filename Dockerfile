FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends docker-cli docker-compose && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && ln -s /usr/bin/docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py .
CMD ["python", "bot.py"]
