

import numpy as np

def accuracy(y_true, y_pred, from_logits=False):
    if from_logits:
        y_pred = np.argmax(y_pred, axis=-1)
    return len(y_pred[y_pred == y_true]) / len(y_pred)