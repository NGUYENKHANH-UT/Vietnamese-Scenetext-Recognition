
def predict_vietocr(detector, image, is_batch=False) -> tuple:
    if is_batch:
      pred, prob = detector.predict_batch(image, return_prob=True)
    else:
      pred, prob = detector.predict(image, return_prob=True)
    return (pred, prob)
