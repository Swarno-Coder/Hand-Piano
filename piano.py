import cv2
import mediapipe as mp
import threading
import pygame

pygame.mixer.init()

# Load piano key sounds
KEYS = [#"C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
        "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
        "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
        "C6"]

SOUNDS = {key: pygame.mixer.Sound(f"soundwav/{key}.wav") for key in KEYS}

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

WHITE_KEY_WIDTH = 60
WHITE_KEY_HEIGHT = 200
BLACK_KEY_WIDTH = 40
BLACK_KEY_HEIGHT = 120

WHITE_KEYS = ["C", "D", "E", "F", "G", "A", "B"]
BLACK_KEYS = ["C#", "D#", "F#", "G#", "A#"]

def generate_piano_keys():
    keys = []
    x_offset = 50  # Starting x position for white keys
    for octave in range(3, 7):
        for note in WHITE_KEYS:
            key_name = f"{note}{octave}"
            keys.append({"name": key_name, "x": x_offset, "color": (255, 255, 255), "pressed": False})
            x_offset += WHITE_KEY_WIDTH

    x_offset = 50 + (WHITE_KEY_WIDTH // 1.5)  # Adjust black key positions
    for octave in range(3, 7):
        for note in BLACK_KEYS:
            if note in ["E#", "B#"]:  # No black keys between E-F and B-C
                continue
            key_name = f"{note}{octave}"
            keys.append({"name": key_name, "x": x_offset, "color": (0, 0, 0), "pressed": False})
            x_offset += WHITE_KEY_WIDTH
        x_offset += WHITE_KEY_WIDTH  # Skip extra spacing for black keys
    return keys

PIANO_KEYS = generate_piano_keys()

def play_note(note):
    if note in SOUNDS:
        threading.Thread(target=SOUNDS[note].play).start()

def detect_pressed_key(x, y):
    for key in PIANO_KEYS:
        if key["color"] == (255, 255, 255):  
            if key["x"] < x < key["x"] + WHITE_KEY_WIDTH and 50 < y < 250:
                return key
        else:
            if key["x"] < x < key["x"] + BLACK_KEY_WIDTH and 50 < y < 170:
                return key
    return None

cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw piano keys
    for key in PIANO_KEYS:
        x1, y1 = key["x"], 50
        x2 = x1 + (WHITE_KEY_WIDTH if key["color"] == (255, 255, 255) else BLACK_KEY_WIDTH)
        y2 = 250 if key["color"] == (255, 255, 255) else 170

        color = (0, 255, 0) if key["pressed"] else key["color"]
        thickness = -1 if key["color"] == (255, 255, 255) else 2  
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(frame, key["name"], (int(x1) + 10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Detect hand and fingers
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id in [8, 12]:  # Index and middle finger tips
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

                    pressed_key = detect_pressed_key(cx, cy)
                    if pressed_key and not pressed_key["pressed"]:
                        pressed_key["pressed"] = True
                        play_note(pressed_key["name"])

    # Reset key states after each frame
    for key in PIANO_KEYS:
        key["pressed"] = False

    cv2.imshow("Virtual Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
