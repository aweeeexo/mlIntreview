import os
import tempfile
import warnings
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from pydub import AudioSegment
import cv2
import mediapipe as mp
import numpy as np
import gigaam
from collections import defaultdict

warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)

# --- Загрузка моделей ---
print("🚀 Загружаю faster-whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ faster-whisper загружен.")

print("🚀 Загружаю GigaAM-Emo model...")
try:
  emotion_model = gigaam.load_model("emo")
  print("✅ GigaAM-Emo загружена.")
except Exception as e:
  print(f"⚠️ Ошибка загрузки GigaAM-Emo: {e}")
  emotion_model = None

# MediaPipe для статической обработки (ИСПРАВЛЕНО: static_image_mode=True)
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

stats_storage = defaultdict(lambda: {
  "blink_count": 0,
  "touch_duration": 0.0,
  "prev_left_dist": None,
  "prev_right_dist": None,
  "touch_active": False,
  "touch_start_time": None
})


# --- Аудио функции ---
def convert_to_wav(input_path, output_path, target_sr=16000):
  audio = AudioSegment.from_file(input_path)
  audio = audio.set_frame_rate(target_sr).set_channels(1)
  audio.export(output_path, format="wav")
  return output_path


def transcribe_audio(file_path):
  segments, _ = whisper_model.transcribe(file_path, language="ru", beam_size=5)
  return " ".join(seg.text for seg in segments).strip()


def analyze_emotions(file_path):
  if emotion_model is None:
    return {}
  try:
    emotions = emotion_model.get_probs(file_path)
    return {k: float(v) for k, v in emotions.items()}
  except Exception as e:
    print(f"Ошибка анализа эмоций: {e}")
    return {}


# --- Обработка кадра ---
def process_frame(image_bytes, question_id):
  try:
    stats = stats_storage[question_id]
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
      return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Обработка
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    # Моргания
    if face_results.multi_face_landmarks:
      landmarks = face_results.multi_face_landmarks[0]
      left_upper = landmarks.landmark[159]
      left_lower = landmarks.landmark[145]
      right_upper = landmarks.landmark[386]
      right_lower = landmarks.landmark[374]
      left_dist = np.hypot(left_upper.x - left_lower.x,
                           left_upper.y - left_lower.y)
      right_dist = np.hypot(right_upper.x - right_lower.x,
                            right_upper.y - right_lower.y)

      if stats["prev_left_dist"] is None or stats["prev_right_dist"] is None:
        stats["prev_left_dist"] = left_dist
        stats["prev_right_dist"] = right_dist
      else:
        left_ratio = left_dist / stats["prev_left_dist"]
        right_ratio = right_dist / stats["prev_right_dist"]
        if left_ratio < 0.5 and right_ratio < 0.5:
          stats["blink_count"] += 1
        stats["prev_left_dist"] = left_dist
        stats["prev_right_dist"] = right_dist

    # Касания лица
    current_touch = False
    if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
      face_landmarks = face_results.multi_face_landmarks[0]
      face_points = [face_landmarks.landmark[1], face_landmarks.landmark[4],
                     face_landmarks.landmark[152]]
      finger_tips = [4, 8, 12, 16, 20]
      for hand_landmarks in hand_results.multi_hand_landmarks:
        for tip in finger_tips:
          tip_point = hand_landmarks.landmark[tip]
          for face_pt in face_points:
            dist = np.hypot(face_pt.x - tip_point.x, face_pt.y - tip_point.y)
            if dist < 0.12:
              current_touch = True
              break
          if current_touch: break
        if current_touch: break

    now = time.time()
    if current_touch and not stats["touch_active"]:
      stats["touch_active"] = True
      stats["touch_start_time"] = now
    elif not current_touch and stats["touch_active"]:
      stats["touch_active"] = False
      stats["touch_duration"] += (now - stats["touch_start_time"])
      stats["touch_start_time"] = None

  except Exception as e:
    print(f"Ошибка в process_frame: {e}")


# --- Эндпоинты ---
@app.route('/process-audio', methods=['POST'])
def process_audio():
  if 'audio' not in request.files:
    return jsonify({'error': 'No audio file'}), 400
  audio_file = request.files['audio']
  with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_in:
    audio_file.save(tmp_in.name)
    tmp_in_path = tmp_in.name
  with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
    tmp_wav_path = tmp_wav.name
  try:
    convert_to_wav(tmp_in_path, tmp_wav_path)
    text = transcribe_audio(tmp_wav_path)
    emotions = analyze_emotions(tmp_wav_path)
    result = {'success': True, 'text': text, 'emotions': emotions}
  except Exception as e:
    result = {'success': False, 'error': str(e)}
  finally:
    if os.path.exists(tmp_in_path): os.unlink(tmp_in_path)
    if os.path.exists(tmp_wav_path): os.unlink(tmp_wav_path)
  return jsonify(result)


@app.route('/process-frame', methods=['POST'])
def process_frame_endpoint():
  try:
    if 'image' not in request.files:
      return jsonify({'error': 'No image'}), 400
    question_id = request.form.get('question_id')
    image_file = request.files['image']
    image_bytes = image_file.read()
    process_frame(image_bytes, question_id)
    return jsonify({'status': 'ok'})
  except Exception as e:
    return jsonify({'error': str(e)}), 500


@app.route('/get-stats', methods=['GET'])
def get_stats():
  question_id = request.args.get('question_id')
  stats = stats_storage.pop(question_id,
                            {"blink_count": 0, "touch_duration": 0.0})
  if stats.get("touch_active", False) and stats.get("touch_start_time"):
    stats["touch_duration"] += (time.time() - stats["touch_start_time"])
  return jsonify({'blinks': stats['blink_count'],
                  'touch_duration': round(stats['touch_duration'], 2)})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000, debug=False)