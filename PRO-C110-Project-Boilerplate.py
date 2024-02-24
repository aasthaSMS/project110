# To Capture Frame
import cv2
# To process image array
import numpy as np
# import the tensorflow modules and load the model
import tensorflow as tf

# Load a pre-trained model (replace 'path/to/model' with the actual path)
model = tf.keras.models.load_model("keras_model.h5")

# Attaching Cam indexed as 0, with the application software
vid = cv2.VideoCapture(0)

# Infinite loop
while(True):
    # Capture the video frame by frame
    ret, frame = vid.read()
    img = cv2.resize(frame, (224, 224))
    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis=0)
    normalize_image = test_image / 255
    prediction = model.predict(normalize_image)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)

    # Define labels
    labels = ["Rock", "Paper","Scissor"]

    # Display the text on the frame
    text = f"Prediction: {labels[predicted_class]}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Print prediction in the console
    print("Prediction:", labels[predicted_class])

    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Quit window with spacebar
    key = cv2.waitKey(1)
    if key == 32:
        break

# Release the camera from the application software
vid.release()

# Close the open window
cv2.destroyAllWindows()
