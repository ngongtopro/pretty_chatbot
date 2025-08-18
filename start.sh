#!/bin/bash
set -e

echo "Waiting for database to be ready..."
python manage.py wait_for_db

echo "Making migrations..."
python manage.py makemigrations

echo "Running database migrations..."
python manage.py migrate --noinput

# Tạo superuser nếu chưa tồn tại
echo "Creating default superuser..."
python manage.py shell <<EOF
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='tinh').exists():
    User.objects.create_superuser('tinh', 'tinh@example.com', '123456')
EOF

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Django development server..."
python manage.py runserver 0.0.0.0:${PORT:-1005}
