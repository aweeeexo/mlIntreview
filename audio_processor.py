import os
import tempfile
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel
import gigaam
from pydub import AudioSegment

# Игнорируем предупреждения
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # разрешаем запросы с любых источников

# --- Загрузка моделей один раз при старте сервера ---
print("🚀 Загружаю модели GigaAM...")
processor = AutoProcessor.from_pretrained("waveletdeboshir/gigaam-ctc",
                                          trust_remote_code=True)
asr_model = AutoModel.from_pretrained("waveletdeboshir/gigaam-ctc",
                                      trust_remote_code=True)
asr_model.eval()

# Загружаем модель эмоций (предполагается, что gigaam.load_model('emo') возвращает объект с методом get_probs)
model_emo = gigaam.load_model('emo')
print("✅ Модели загружены.")


def convert_to_wav(input_path, output_path, target_sr=16000):
  """
    Конвертирует любой аудиофайл (webm, mp4, ogg и т.д.) в WAV с заданной частотой.
    Использует pydub (требует ffmpeg).
    """
  audio = AudioSegment.from_file(input_path)
  audio = audio.set_frame_rate(target_sr).set_channels(1)  # моно, 16 кГц
  audio.export(output_path, format="wav")
  return output_path


def transcribe_audio(file_path):
  """Распознаёт речь из WAV-файла и возвращает текст."""
  wav, orig_sr = torchaudio.load(file_path)
  if orig_sr != 16000:
    wav = torchaudio.functional.resample(wav, orig_sr, 16000)
  input_features = processor(wav[0], sampling_rate=16000, return_tensors="pt")
  with torch.no_grad():
    logits = asr_model(**input_features).logits
  greedy_ids = logits.argmax(dim=-1)
  transcription = processor.batch_decode(greedy_ids)[0]
  return transcription.strip()


def analyze_emotions(file_path):
  """Возвращает словарь с вероятностями эмоций."""
  return model_emo.get_probs(file_path)


@app.route('/process-audio', methods=['POST'])
def process_audio():
  if 'audio' not in request.files:
    return jsonify({'error': 'No audio file'}), 400

  audio_file = request.files['audio']

  # Сохраняем входящий файл во временный файл (с оригинальным расширением)
  with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_in:
    audio_file.save(tmp_in.name)
    tmp_in_path = tmp_in.name

  # Создаём временный WAV-файл для моделей
  with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
    tmp_wav_path = tmp_wav.name

  try:
    # Конвертируем в WAV (16 кГц, моно)
    convert_to_wav(tmp_in_path, tmp_wav_path, target_sr=16000)

    # Распознавание речи
    transcription = transcribe_audio(tmp_wav_path)

    # Анализ эмоций (можно передать WAV, если модель поддерживает, или конвертировать ещё раз)
    emotions = analyze_emotions(tmp_wav_path)

    result = {
      'success': True,
      'text': transcription,
      'emotions': emotions
    }
  except Exception as e:
    result = {
      'success': False,
      'error': str(e)
    }
  finally:
    # Удаляем временные файлы
    os.unlink(tmp_in_path)
    os.unlink(tmp_wav_path)

  return jsonify(result)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=True)