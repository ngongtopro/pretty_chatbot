#Hướng dẫn chạy window
# 1. Tạo virtual environment
python -m venv venv

# 2. Kích hoạt venv (Windows)
.\venv\Scripts\activate

# 2. Kích hoạt venv (ubuntu)
source .\venv\Scripts\activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy server FastAPI
uvicorn embedding_api:app --host 0.0.0.0 --port 8000
