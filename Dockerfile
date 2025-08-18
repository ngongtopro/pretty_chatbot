FROM python:3.13-slim AS builder
# python:3.13-alpine: build lâu hơn, dễ lỗi nhưng igmage nhẹ
# python:3.13-slim: dễ tương thích với psycopg2, .... build nhanh hơn. image to hơn

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cài pg_config và tool build, fix lỗi của psycopg2, nếu ko sẽ phải dùng psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    build-essential \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY heavy_requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r heavy_requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy và set permission cho các script
COPY wait-for-it.sh /wait-for-it.sh
COPY start.sh /start.sh
RUN chmod +x /wait-for-it.sh /start.sh

ARG PORT=1005

ENV PORT=${PORT}

# Expose the Django port
EXPOSE ${PORT}

CMD ["/start.sh"]
# Các option khác:
#CMD ["/wait-for-it.sh", "db", "5432", "python", "manage.py", "runserver", "0.0.0.0:1005"]
#CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:${PORT}"]