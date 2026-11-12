# mlIntreview

Python-сервис для анализа ответов кандидата на видеоинтервью. Принимает аудио и кадры с веб-камеры и возвращает:

- **транскрипцию речи** (faster-whisper, русский язык);
- **эмоции в голосе** (GigaAM-Emo);
- **число морганий** и **время касания лица руками** (MediaPipe Face Mesh + Hands).

Сервис рассчитан на работу в связке с основным бэкендом (Java) через общую Docker-сеть `ai-career-network`.

## Структура

| Файл | Назначение |
|---|---|
| `app.py` | Основной Flask-сервер (порт 5000): аудио и видеоаналитика |
| `audio_processor.py` | Альтернативный аудиосервер на GigaAM-CTC (transformers) |
| `video_processor.py` | Локальный прототип детекции морганий и касаний лица с веб-камеры |
| `Dockerfile`, `docker-compose.yml` | Сборка и запуск в контейнере |

## API

### `POST /process-audio`

`multipart/form-data`, поле `audio` — аудиофайл (webm, ogg, mp4, wav…).

```json
{ "success": true, "text": "распознанный текст", "emotions": { "angry": 0.01, "sad": 0.02, "neutral": 0.9, "positive": 0.07 } }
```

### `POST /process-frame`

`multipart/form-data`: `image` — кадр (jpeg/png), `question_id` — идентификатор вопроса. Накапливает статистику по вопросу.

```json
{ "status": "ok" }
```

### `GET /get-stats?question_id=<id>`

Возвращает накопленную статистику по вопросу и сбрасывает её.

```json
{ "blinks": 12, "touch_duration": 3.45 }
```

## Запуск

### Docker

Сеть и volume объявлены как внешние, их нужно создать заранее:

```bash
docker network create ai-career-network
docker volume create my_global_code_storage
docker compose up --build
```

### Локально

Требуется Python 3.10 и `ffmpeg` в `PATH`.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

При первом запуске скачиваются веса моделей Whisper и GigaAM.
