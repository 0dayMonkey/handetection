# HandTrackingModule.py
"""
Module d'ingénierie pour la détection des mains et l'analyse de gestes.

Ce module fournit une classe, HandDetector, qui encapsule le modèle MediaPipe
pour détecter les points de repère de la main, appliquer un lissage temporel
pour la stabilité, et calculer des métriques de gestes.

Version: 4.0
Auteur: JPL/NASA HMI Division
"""

import cv2
import mediapipe as mp
import math
import numpy as np
from typing import List, Optional, Tuple, Any

class HandDetector:
    """
    Encapsule la détection des mains et l'analyse des gestes via MediaPipe.
    
    Fournit des méthodes pour trouver les mains dans une image, extraire et
    lisser les positions des points de repère, et interpréter des gestes
    de base comme le nombre de doigts levés ou la distance de pincement.
    """
    
    THUMB_TIP_ID: int = 4
    INDEX_FINGER_TIP_ID: int = 8
    MIDDLE_FINGER_TIP_ID: int = 12
    RING_FINGER_TIP_ID: int = 16
    PINKY_TIP_ID: int = 20
    INDEX_FINGER_PIP_ID: int = 6
    MIDDLE_FINGER_PIP_ID: int = 10
    RING_FINGER_PIP_ID: int = 14
    PINKY_PIP_ID: int = 18

    def __init__(self,
                 max_hands: int = 1,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7,
                 smoothing_factor: float = 0.6) -> None:
        """
        Initialise le détecteur de main.

        Args:
            max_hands: Nombre maximum de mains à détecter.
            detection_confidence: Seuil de confiance minimal pour la détection.
            tracking_confidence: Seuil de confiance minimal pour le suivi.
            smoothing_factor: Facteur de lissage pour l'EMA (0: max lissage, 1: pas de lissage).
        """
        assert 0 <= smoothing_factor <= 1, "Le facteur de lissage doit être entre 0 et 1."
        
        self.smoothing_factor: float = smoothing_factor
        self.prev_landmarks: Optional[np.ndarray] = None
        
        self.mp_hands: Any = mp.solutions.hands
        self.hands: Any = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw: Any = mp.solutions.drawing_utils
        self.results: Any = None
        self.tip_ids: List[int] = [
            self.THUMB_TIP_ID, self.INDEX_FINGER_TIP_ID, self.MIDDLE_FINGER_TIP_ID,
            self.RING_FINGER_TIP_ID, self.PINKY_TIP_ID
        ]

    def find_hands(self, image: np.ndarray) -> None:
        """
        Traite une image pour détecter les mains.
        
        Les résultats de la détection sont stockés dans l'attribut `self.results`.

        Args:
            image: L'image (tableau NumPy BGR) à analyser.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        self.results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

    def find_position(self, image: np.ndarray, hand_index: int = 0) -> List[List[int]]:
        """
        Extrait et lisse les coordonnées des points de repère pour une main.

        Le lissage par moyenne mobile exponentielle (EMA) est appliqué pour
        stabiliser les coordonnées des points, réduisant ainsi la gigue.

        Args:
            image: L'image sur laquelle les dimensions sont basées.
            hand_index: L'index de la main à traiter (0 par défaut).

        Returns:
            Une liste de listes [id, x, y] pour chaque point de repère de la
            main détectée. Retourne une liste vide si aucune main n'est trouvée.
        """
        raw_landmark_list = []
        if self.results.multi_hand_landmarks and hand_index < len(self.results.multi_hand_landmarks):
            target_hand = self.results.multi_hand_landmarks[hand_index]
            height, width, _ = image.shape
            
            for landmark_id, landmark in enumerate(target_hand.landmark):
                px, py = int(landmark.x * width), int(landmark.y * height)
                raw_landmark_list.append([landmark_id, px, py])

            current_landmarks_np = np.array(raw_landmark_list, dtype=np.float32)

            if self.prev_landmarks is not None:
                smoothed_landmarks_np = (self.smoothing_factor * current_landmarks_np) + \
                                        ((1 - self.smoothing_factor) * self.prev_landmarks)
                self.prev_landmarks = smoothed_landmarks_np
            else:
                self.prev_landmarks = current_landmarks_np
            
            return [[int(lm[0]), int(lm[1]), int(lm[2])] for lm in self.prev_landmarks]
        
        self.prev_landmarks = None
        return []

    def get_open_fingers_count(self, landmark_list: List[List[int]]) -> int:
        """
        Compte le nombre de doigts levés en se basant sur la liste des points de repère.

        Args:
            landmark_list: La liste des points de repère d'une main.

        Returns:
            Le nombre de doigts considérés comme ouverts (de 0 à 5).
        """
        if not landmark_list:
            return 0
            
        fingers_open_status = []
        
        is_thumb_open = landmark_list[self.tip_ids[0]][1] > landmark_list[self.tip_ids[0] - 1][1]
        fingers_open_status.append(1 if is_thumb_open else 0)

        pip_joint_ids = [
            self.INDEX_FINGER_PIP_ID, self.MIDDLE_FINGER_PIP_ID,
            self.RING_FINGER_PIP_ID, self.PINKY_PIP_ID
        ]
        for i in range(1, 5):
            is_finger_open = landmark_list[self.tip_ids[i]][2] < landmark_list[pip_joint_ids[i-1]][2]
            fingers_open_status.append(1 if is_finger_open else 0)
            
        return sum(fingers_open_status)
        
    def get_pinch_distance(self, landmark_list: List[List[int]]) -> Optional[float]:
        """
        Calcule la distance euclidienne entre le bout du pouce et de l'index.

        Args:
            landmark_list: La liste des points de repère d'une main.

        Returns:
            La distance en pixels, ou None si les points ne sont pas disponibles.
        """
        if len(landmark_list) > max(self.THUMB_TIP_ID, self.INDEX_FINGER_TIP_ID):
            p1 = landmark_list[self.THUMB_TIP_ID][1:]
            p2 = landmark_list[self.INDEX_FINGER_TIP_ID][1:]
            return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return None

    def draw_hand_landmarks(self, image: np.ndarray) -> None:
        """
        Dessine le squelette et les points de repère sur l'image.

        Args:
            image: L'image (tableau NumPy BGR) sur laquelle dessiner.
        """
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)