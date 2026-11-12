FROM python:3.10-slim

# Обновленный список системных библиотек
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ЭТА СТРОКА ВЫВЕДЕТ ЛОГИ ПИТОНА В РЕАЛЬНОМ ВРЕМЕНИ
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]