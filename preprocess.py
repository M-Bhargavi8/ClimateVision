"""Image loading shared by training and the Streamlit app.

Both sides must turn a photo into exactly the same array, or the model sees
different inputs at serving time than it was trained on. Keeping that in one
function is what guarantees it.
"""

import cv2
import numpy as np

IMAGE_SIZE = 224


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode image bytes to a 224x224 RGB uint8 array, or None if unreadable.

    IMREAD_COLOR always yields three channels, so a PNG with transparency or a
    greyscale photo becomes plain RGB instead of crashing the model with a
    4-channel or 1-channel input.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # BGR, alpha dropped
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # INTER_AREA is the right filter for shrinking photos: it averages pixels
    # instead of skipping them, so fine texture (rain streaks) survives.
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def load_image(path: str) -> np.ndarray | None:
    with open(path, "rb") as f:
        return decode_image(f.read())
