import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Initialize the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    _, frame = webcam.read()

    # Preprocess the frame for the model
    preprocessed_frame = preprocess_frame(frame)

    # Use the model to make a prediction on the frame
    prediction = model.predict(preprocessed_frame)

    # Display the prediction on the frame
    display_prediction(frame, prediction)

    # Check for user input to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
webcam.release()
cv2.destroyAllWindows()