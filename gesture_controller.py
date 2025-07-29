# gesture_controller.py
"""
Application principale de contrôle du système par gestes de la main.

Ce script utilise la webcam pour détecter les gestes de la main et contrôler
le volume audio du système (Windows uniquement) ou la luminosité de l'écran.
Le changement de mode de contrôle (volume/luminosité) s'effectue en
fermant la main pendant un court instant.

Version: 4.2
Auteur: JPL/NASA HMI Division
"""

import cv2
import time
import numpy as np
import platform
from typing import Optional, Tuple, Any, Dict, List

from HandTrackingModule import HandDetector

IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import screen_brightness_control as sbc

# --- Constantes de configuration ---
CAM_WIDTH: int = 1280
CAM_HEIGHT: int = 720
HAND_DISTANCE_MIN: int = 30
HAND_DISTANCE_MAX: int = 250
BRIGHTNESS_MIN: int = 0
BRIGHTNESS_MAX: int = 100
GESTURE_CONFIRM_FRAMES: int = 5

# --- Constantes UI ---
UI_BAR_POS_X: int = 50
UI_BAR_HEIGHT_MAX: int = 400
UI_BAR_HEIGHT_MIN: int = 150
UI_BAR_WIDTH: int = 35
UI_BAR_COLOR: Tuple[int, int, int] = (255, 255, 0)
UI_TEXT_COLOR_MAIN: Tuple[int, int, int] = (255, 255, 0)
UI_TEXT_COLOR_MODE: Tuple[int, int, int] = (0, 0, 255)

def initialize_system() -> Tuple[cv2.VideoCapture, HandDetector, Optional[Any], Optional[Tuple[float, float]]]:
    """
    Initialise la caméra, le détecteur de main et le contrôle du volume.

    Returns:
        Un tuple contenant l'objet de capture vidéo, le détecteur de main,
        l'interface de contrôle du volume (ou None), et la plage de volume
        système (ou None).
    """
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Erreur critique : Impossible d'accéder à la webcam."
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    detector = HandDetector(smoothing_factor=0.6)
    
    volume_control = None
    volume_range_system = None
    if IS_WINDOWS:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_control = cast(interface, POINTER(IAudioEndpointVolume))
        volume_range_system = volume_control.GetVolumeRange()[:2]
        
    return cap, detector, volume_control, volume_range_system

def update_control_value(mode: str, distance: float, vol_ctrl: Optional[Any], vol_range: Optional[Tuple[float, float]]) -> None:
    """
    Met à jour la valeur de contrôle (volume ou luminosité) en fonction de la distance.
    """
    if mode == "Volume" and IS_WINDOWS and vol_ctrl is not None and vol_range is not None:
        control_value = np.interp(distance, [HAND_DISTANCE_MIN, HAND_DISTANCE_MAX], vol_range)
        vol_ctrl.SetMasterVolumeLevel(control_value, None)
    elif mode == "Luminosite":
        control_value = np.interp(distance, [HAND_DISTANCE_MIN, HAND_DISTANCE_MAX], [BRIGHTNESS_MIN, BRIGHTNESS_MAX])
        sbc.set_brightness(int(control_value))

def process_gestures(detector: HandDetector, landmark_list: List[List[int]], app_state: Dict[str, Any]) -> None:
    """
    Analyse les gestes selon une logique binaire : main fermée ou main de contrôle.
    """
    if not landmark_list:
        app_state["gesture_counter"] = 0
        return

    open_fingers = detector.get_open_fingers_count(landmark_list)
    is_hand_closed = open_fingers < 2

    # État 1: Geste de commutation (main fermée)
    if is_hand_closed:
        app_state["gesture_counter"] += 1
        if app_state["gesture_counter"] == GESTURE_CONFIRM_FRAMES:
            app_state["current_mode"] = "Luminosite" if app_state["current_mode"] == "Volume" else "Volume"
    
    # État 2: Geste de contrôle (toute autre configuration)
    else:
        app_state["gesture_counter"] = 0
        distance = detector.get_pinch_distance(landmark_list)
        if distance is not None:
            app_state["last_distance"] = distance
            update_control_value(app_state["current_mode"], distance, app_state["volume_control"], app_state["volume_range"])

def draw_ui(image: np.ndarray, mode: str, distance: float) -> None:
    """
    Dessine l'interface utilisateur sur l'image.
    """
    cv2.putText(image, f"Mode: {mode}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, UI_TEXT_COLOR_MODE, 3)
    
    bar_height = np.interp(distance, [HAND_DISTANCE_MIN, HAND_DISTANCE_MAX], [UI_BAR_HEIGHT_MAX, UI_BAR_HEIGHT_MIN])
    bar_percentage = np.interp(distance, [HAND_DISTANCE_MIN, HAND_DISTANCE_MAX], [0, 100])
    
    cv2.rectangle(image, (UI_BAR_POS_X, UI_BAR_HEIGHT_MIN), (UI_BAR_POS_X + UI_BAR_WIDTH, UI_BAR_HEIGHT_MAX), UI_BAR_COLOR, 3)
    cv2.rectangle(image, (UI_BAR_POS_X, int(bar_height)), (UI_BAR_POS_X + UI_BAR_WIDTH, UI_BAR_HEIGHT_MAX), UI_BAR_COLOR, cv2.FILLED)
    cv2.putText(image, f'{int(bar_percentage)} %', (UI_BAR_POS_X - 5, UI_BAR_HEIGHT_MAX + 35), cv2.FONT_HERSHEY_PLAIN, 2, UI_TEXT_COLOR_MAIN, 2)

def run_application() -> None:
    """Lance la boucle principale de l'application."""
    cap, detector, vol_ctrl, vol_range = initialize_system()
    
    app_state = {
        "current_mode": "Volume",
        "gesture_counter": 0,
        "last_distance": HAND_DISTANCE_MIN,
        "volume_control": vol_ctrl,
        "volume_range": vol_range
    }

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.flip(image, 1)

        detector.find_hands(image)
        landmark_list = detector.find_position(image)
        
        process_gestures(detector, landmark_list, app_state)
        
        detector.draw_hand_landmarks(image)
        draw_ui(image, app_state["current_mode"], app_state["last_distance"])
        
        cv2.imshow("JPL - Gesture Control v4.2", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_application()