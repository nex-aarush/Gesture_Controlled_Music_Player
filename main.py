import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path
import pygame

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

sounds_swaras = Path(__file__).resolve().parent / "sounds_swaras"

sounds_mapping = {
    1: sounds_swaras/"1_Sa.mp3",  
    2: sounds_swaras/"2_Re.mp3",
    3: sounds_swaras/"3_Ga.mp3",
    4: sounds_swaras/"4_Ma.mp3",
    5: sounds_swaras/"5_Pa.mp3",
    6: sounds_swaras/"6_Dha.mp3",
    7: sounds_swaras/"7_Ni.mp3",
    8: sounds_swaras/"8_Sa_upper.mp3",
    10: sounds_swaras/"10_Actual_Song.mp3"
}

swara_names = {
    1: "Sa",
    2: "Re",
    3: "Ga",
    4: "Ma",
    5: "Pa",
    6: "Dha",
    7: "Ni",
    8: "Sa (upper)",
    10: "An Actual Song :)"
}

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0,255,0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0,0,255), -1)

def count_fingers(hand_landmarks,label):
    count = 0
    if hand_landmarks[8].y < hand_landmarks[6].y:
        count += 1
    if hand_landmarks[12].y < hand_landmarks[10].y:
        count += 1
    if hand_landmarks[16].y < hand_landmarks[14].y:
        count += 1
    if hand_landmarks[20].y < hand_landmarks[18].y:
        count += 1
    
    thumb_tip_x = hand_landmarks[4].x
    thumb_joint_x = hand_landmarks[3].x
    if label == "Left":
        if thumb_tip_x < thumb_joint_x:
            count += 1
    elif label == "Right":
        if thumb_tip_x > thumb_joint_x:
            count += 1
    
    return count

start_time = time.time()

MODEL_PATH = Path(__file__).resolve().parent/ "hand_landmarker.task"
base_options = mp_python.BaseOptions(model_asset_path = str(MODEL_PATH))
options = mp_vision.HandLandmarkerOptions(
    base_options = base_options,
    running_mode = mp_vision.RunningMode.VIDEO,
    num_hands = 2,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

pygame.mixer.init()
previous_count = 0

capture = cv2.VideoCapture(0)
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        total_count = 0
        for i,hand_landmarks in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name
            total_count += count_fingers(hand_landmarks, label)
            draw_landmarks(frame, hand_landmarks)

        if total_count in sounds_mapping:
            sound_path = sounds_mapping[total_count]
            swara_name = swara_names[total_count]
        else:
            sound_path = None
            swara_name = "Silence"
        
        if total_count != previous_count:
            if sound_path is not None:
                pygame.mixer.music.load(str(sound_path))
                pygame.mixer.music.play()
            else:
                pygame.mixer.music.stop()
        previous_count = total_count

        cv2.putText(frame,f"Hands: {len(result.hand_landmarks)}",(10, 90),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,)
        cv2.putText(frame,f"Number of fingers: {total_count} ",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,)
        cv2.putText(frame,f"Swara: {swara_name}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,)

    cv2.imshow("Swara Player - type q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
landmarker.close()
pygame.mixer.quit()