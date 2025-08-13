#!/bin/bash
set -e

echo "Waiting for database to be ready..."
python manage.py wait_for_db

echo "Making migrations for chat app..."
python manage.py makemigrations chat

echo "Running database migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting Django development server..."
python manage.py runserver 0.0.0.0:${PORT:-1005}
