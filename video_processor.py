import math

import cv2
import mediapipe as mp



# Инициализируем нужные модули MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Создаем экземпляр F=;e Mesh
# max_num_faces: сколько лиц искать
# refine_landmarks: включаем доп. точки для глаз и губ для большей точности
# min_detection_confidence: порог уверенности для обнаружения лица
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
# Захватываем видео с веб-камеры (0 - это обычно встроенная камера)
cap = cv2.VideoCapture(0)
dist = []
blink_cnt = 0
blink_status = "open"
facetouch_time = 0
FINGERTIP_INDICES = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]
while cap.isOpened():
    # Читаем кадр с камеры
    success, image = cap.read()
    if not success:
        print("Не удалось получить кадр с камеры.")
        continue

    # Для повышения производительности делаем изображение "только для чтения"
    image.flags.writeable = False
    # Переводим изображение из BGR (OpenCV) в RGB (MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обрабатываем кадр и находим ориентиры лица
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)
    # Снова делаем изображение доступным для записи, чтобы рисовать на нем
    image.flags.writeable = True
    # Переводим изображение обратно в BGR для отображения через OpenCV
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    # Рисуем сетку на лице, если оно найдено
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # FACEMESH_TESSELATION - это сама сетка (мелкие треугольники)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None, # Не рисуем сами точки, только соединения
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # FACEMESH_CONTOURS - основные контуры лица (глаза, брови, губы)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None, # Не рисуем сами точки
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            point_Hgh_left_eye = face_landmarks.landmark[159]
            pixel_x_hle = int(point_Hgh_left_eye.x * image_width)
            pixel_y_hle = int(point_Hgh_left_eye.y * image_height)

            point_low_left_eye = face_landmarks.landmark[145]
            pixel_x_lle = int(point_low_left_eye.x * image_width)
            pixel_y_lle = int(point_low_left_eye.y * image_height)

            dist_now = ((pixel_x_hle-pixel_x_lle)**2 + (pixel_y_hle-pixel_y_lle)**2)**0.5

            dist.append(dist_now)

            if not(dist[0] > dist_now*2 and blink_status == "open"):
                blink_status = "open"

            if dist[0] > dist_now*2 and blink_status == "open":
                blink_cnt+=1
                blink_status = "blink"

    print(blink_cnt//3, blink_status)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    if face_results.multi_face_landmarks and hand_results.multi_hand_landmarks:
        anchor_point_face = face_results.multi_face_landmarks[0].landmark[1]
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for tip_index in FINGERTIP_INDICES:
                fingertip = hand_landmarks.landmark[tip_index]
                distance = math.sqrt(
                    (anchor_point_face.x - fingertip.x) ** 2 + (
                        anchor_point_face.y - fingertip.y) ** 2)
                if distance < 0.1:
                    facetouch_time+=1
                    print(facetouch_time // 20)
                    text_x = int(fingertip.x * image_width)
                    text_y = int(fingertip.y * image_height)
                    cv2.putText(image, "Finger Near Face", (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    anchor_pixel = (int(anchor_point_face.x * image_width),
                                    int(anchor_point_face.y * image_height))
                    fingertip_pixel = (text_x, text_y)
                    cv2.line(image, anchor_pixel, fingertip_pixel, (0, 255, 0),
                             2)
                    break

    # Отражаем изображение по горизонтали для удобного "зеркального" вида
    flipped_image = cv2.flip(image, 1)

    # Показываем результат в окне
    cv2.imshow('MediaPipe Face Mesh', flipped_image)

    # Выход из цикла по нйжатию клавиши 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break



# Освобождаем камеру и закрываем все окна
cap.release()
cv2.destroyAllWindows()

