import cv2
import mediapipe as mp
import numpy as np
from music21 import note, midi, stream
import pygame

# Initialize pygame mixer for real-time playback
pygame.mixer.init()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define piano keys (C4 to C5 including sharps)
KEYS = ['C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5']
WHITE_KEYS = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
BLACK_KEYS = ['C#4', 'D#4', 'F#4', 'G#4', 'A#4']

# Key Positions (White and Black)
WHITE_KEY_POS = [(i * 100, 0, (i + 1) * 100, 100) for i in range(len(WHITE_KEYS))]
BLACK_KEY_POS = [(i * 100 + 75, 0, i * 100 + 125, 50) for i in range(len(BLACK_KEYS))]

# Load MIDI sounds dynamically using music21
def play_note(note_name):
    n = note.Note(note_name)
    s = stream.Stream()
    s.append(n)
    sp = midi.realtime.StreamPlayer(s)
    sp.play()

# Open Camera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw piano keys (White) outline
    for i, (x1, y1, x2, y2) in enumerate(WHITE_KEY_POS):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, WHITE_KEYS[i], (x1 + 30, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Draw piano keys (Black) outline
    for i, (x1, y1, x2, y2) in enumerate(BLACK_KEY_POS):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, BLACK_KEYS[i], (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Track index finger tip (Landmark 8)
                if id == 8:
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

                    # Check if finger is pressing a black key first (priority)
                    for i, (x1, y1, x2, y2) in enumerate(BLACK_KEY_POS):
                        if x1 < cx < x2 and y1 < cy < y2:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, BLACK_KEYS[i], (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            play_note(BLACK_KEYS[i])

                    # Check if finger is pressing a white key (if not on black)
                    for i, (x1, y1, x2, y2) in enumerate(WHITE_KEY_POS):
                        if x1 < cx < x2 and y1 < cy < y2:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, WHITE_KEYS[i], (x1 + 30, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            play_note(WHITE_KEYS[i])

    cv2.imshow("Virtual Piano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
