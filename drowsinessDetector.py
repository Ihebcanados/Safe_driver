# Importation de la librairie OpenCV pour la capture vidéo et le traitement d'images
import cv2

# Numpy pour les opérations numériques (calculs, distances…)
import numpy as np

# Dlib pour la détection du visage + détection des 68 points du visage (landmarks)
import dlib

# Librairie utilitaire pour convertir les landmarks de dlib → tableau numpy
from imutils import face_utils

# Communication série avec Arduino / Proteus
import serial

# Ouverture du port série COM1 avec un baudrate de 9600 pour envoyer les alertes
s = serial.Serial('COM1', 9600)

# Démarrage du flux caméra (0 = webcam par défaut)
cap = cv2.VideoCapture(0)

# Initialisation du détecteur de visage basé sur HOG (dlib)
hog_face_detector = dlib.get_frontal_face_detector()

# Chargement du modèle des 68 points du visage
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Compteurs pour suivre l’état du conducteur (éviter les fausses alertes)
sleep = 0
drowsy = 0
active = 0

# Texte + couleur affichés sur l’écran
status = ""
color = (0, 0, 0)

# ---------------------------------------------------------
# Fonction pour calculer la distance euclidienne entre 2 points
# Utilisée pour mesurer l’ouverture/fermeture des yeux
# ---------------------------------------------------------
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# -------------------------------------------------------------------
# Fonction Eye Aspect Ratio (EAR)
# Permet de déterminer si l’œil est ouvert / semi-ouvert / fermé
# a,f = coins de l’œil
# b,c,d,e = points verticaux
# -------------------------------------------------------------------
def blinked(a, b, c, d, e, f):
    # Distance verticale (paupière supérieure + inférieure)
    up = compute(b, d) + compute(c, e)
    # Distance horizontale (largeur de l’œil)
    down = compute(a, f)

    # Ratio (EAR)
    ratio = up / (2.0 * down)

    # Définition:
    # > 0.25 → œil ouvert
    # 0.21–0.25 → semi-ouvert (somnolence)
    # < 0.21 → fermé (sommeil)
    if ratio > 0.25:
        return 2     # œil bien ouvert
    elif ratio > 0.21 and ratio <= 0.25:
        return 1     # moitié ouvert = somnolent
    else:
        return 0     # fermé = en train de dormir


# ---------------------------------------------------------
# Boucle principale (lecture image par image de la caméra)
# ---------------------------------------------------------
while True:
        # Capture d’une image depuis la caméra
        _, frame = cap.read()

        # Conversion en niveaux de gris (plus rapide et plus précis pour la détection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection de visages dans l’image
        faces = hog_face_detector(gray)
        face_frame = frame.copy()
        # Parcours de chaque visage détecté
        for face in faces:
                # Récupération des coordonnées du visage
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                # Copie du frame original pour dessiner les landmarks
                face_frame = frame.copy()

                # Dessiner un rectangle autour du visage détecté
                cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prédire les 68 points du visage
                landmarks = predictor(gray, face)

                # Conversion → tableau numpy manipulable
                landmarks = face_utils.shape_to_np(landmarks)

                # Analyse de l’œil gauche et droit pour détecter le clignement
                left_blink = blinked(
                    landmarks[36], landmarks[37], landmarks[38],
                    landmarks[41], landmarks[40], landmarks[39]
                )
                right_blink = blinked(
                    landmarks[42], landmarks[43], landmarks[44],
                    landmarks[47], landmarks[46], landmarks[45]
                )

                # ---------------------------------------------------------
                # Analyse de l’état du conducteur à partir des yeux
                # ---------------------------------------------------------
                if left_blink == 0 or right_blink == 0:
                        # Œil fermé → peut être en train de dormir
                        sleep += 1
                        drowsy = 0
                        active = 0

                        # Si fermé pendant +6 frames → valide "Sleep"
                        if sleep > 6:
                                s.write(b's')                 # envoyer alerte à Arduino
                                status = "Driver SLEEPING"
                                color = (0, 0, 255)

                elif left_blink == 1 or right_blink == 1:
                        # Œil semi-fermé → somnolence
                        sleep = 0
                        active = 0
                        drowsy += 1

                        # Si somnolent pendant +6 frames → valide "Drowsy"
                        if drowsy > 6:
                                s.write(b'd')
                                status = "Driver Drowsy"
                                color = (0, 0, 255)

                else:
                        # Yeux ouverts → conducteur actif
                        drowsy = 0
                        sleep = 0
                        active += 1

                        # Stabilisation pendant +6 frames → valide "Active"
                        if active > 6:
                                s.write(b'a')
                                status = "Driver Active"
                                color = (0, 255, 0)

                # Affichage du statut sur la vidéo
                cv2.putText(frame, status, (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                # Affichage des 68 landmarks (petits points blancs)
                for n in range(0, 68):
                        (x, y) = landmarks[n]
                        cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        # Affichage du flux principal + fenêtre des landmarks
        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)

        # Touche Échap pour quitter
        key = cv2.waitKey(1)
        if key == 27:
                break
