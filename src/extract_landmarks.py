import numpy as np

def extract_landmarks(results):
    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        keypoints = np.array([
            [lm.x, lm.y, lm.z]
            for lm in hand.landmark
        ])

    else:
        keypoints = np.zeros((21,3))

    return keypoints