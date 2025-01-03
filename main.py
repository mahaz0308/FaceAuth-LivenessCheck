import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time

def rotate_and_detect_faces(image):
    # Check for faces in the original orientation
    face_locations = face_recognition.face_locations(image)
    if face_locations:
        return image, face_locations  # Return image and detected faces

    # Rotate the image by 90, 180, or 270 degrees and check each one
    for angle in [90, 180, 270]:
        rotated_image = np.array(Image.fromarray(image).rotate(angle, expand=True))  # Rotate image
        rotated_face_locations = face_recognition.face_locations(rotated_image)
        if rotated_face_locations:
            print(f"Face detected after rotating by {angle} degrees")
            return rotated_image, rotated_face_locations

    # If no faces detected after all rotations, return None
    print("No face detected in the image after rotation.")
    return None

def liveness_check():
    # Use the webcam to detect if the user blinks or moves their face
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return False

    print("Please blink or move your face to verify liveness.")
    
    # Initialize the face cascade for detecting faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()  # Start timer to check if user is taking too long

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Convert the image to grayscale for easier face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If faces are detected, draw rectangles around detected faces
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle around the face
            # Display the frame with the detected face
            cv2.imshow('Liveness Check - Move or Blink', frame)

            # Wait for a few seconds or wait for the user to press a key (e.g., 'c' to continue)
            print("Liveness detected! Please look at the camera for a few seconds.")
            key = cv2.waitKey(5000) & 0xFF  # Wait for 5 seconds to let the user see the rectangle drawn

            if key == ord('c'):  # If the user presses 'c', continue
                break  # Exit the loop once liveness is confirmed
        else:
            # If no face is detected, continue showing the webcam feed
            cv2.imshow('Liveness Check - Move or Blink', frame)

        # Allow the user to press 'q' to quit if they wish
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting liveness check.")
            break

        # Timeout if the liveness check takes too long (e.g., 30 seconds)
        if time.time() - start_time > 10:
            print("Liveness check timed out.")
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    return len(faces) > 0  # Return True if faces were detected, else False

def capture_selfie(save_path="selfie_captured.png"):
    # Initialize the webcam to capture a selfie after liveness detection
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return None

    print("Selfie capture. Please position yourself.")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Display the resulting frame
        cv2.imshow('Selfie Capture', frame)

        # Wait for the user to press a key (e.g., 's' for save and quit)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Save the captured image
            cv2.imwrite(save_path, frame)
            print(f"Selfie saved to {save_path}")
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

    return save_path

def compare_faces(cnic_image_path, selfie_image_path):
    # Load and detect faces in CNIC and selfie images with rotation handling
    cnic_image = face_recognition.load_image_file(cnic_image_path)
    selfie_image = face_recognition.load_image_file(selfie_image_path)

    # Detect faces in CNIC image with rotation handling
    cnic_image, cnic_face_locations = rotate_and_detect_faces(cnic_image)
    if cnic_image is None:
        print("No face detected in CNIC image.")
        return

    # Detect faces in selfie image with rotation handling
    selfie_image, selfie_face_locations = rotate_and_detect_faces(selfie_image)
    if selfie_image is None:
        print("No face detected in selfie image.")
        return

    # Generate face encodings for all detected faces
    cnic_face_encodings = face_recognition.face_encodings(cnic_image, cnic_face_locations)
    selfie_face_encodings = face_recognition.face_encodings(selfie_image, selfie_face_locations)

    # If faces are detected in both images, compare them
    if cnic_face_encodings and selfie_face_encodings:
        # For simplicity, comparing the first face detected in each image
        results = face_recognition.compare_faces(cnic_face_encodings, selfie_face_encodings[0])
        
        # Calculate the face distance (similarity score)
        face_distances = face_recognition.face_distance(cnic_face_encodings, selfie_face_encodings[0])
        
        # Show results
        if any(results):
            print("The faces match!")
            print(f"Similarity score (lower is better): {face_distances[0]:.4f}")
            if face_distances[0] < 0.6:
                print("The faces are highly similar based on the similarity score.")
            else:
                print("The faces match but the similarity score suggests a moderate difference.")
        else:
            print("The faces do not match.")
            print(f"Similarity score: {face_distances[0]:.4f}")
            if face_distances[0] > 0.6:
                print("The similarity score is high, indicating that the faces are quite different.")
            else:
                print("The similarity score is relatively low, suggesting a closer match, but other factors might cause a mismatch.")
    else:
        print("No face encodings found for comparison.")

    # Visualize faces with bounding boxes for CNIC image
    cnic_image_pil = Image.fromarray(cnic_image)  # Convert to PIL Image
    draw_cnic = ImageDraw.Draw(cnic_image_pil)

    for (top, right, bottom, left) in cnic_face_locations:
        draw_cnic.rectangle([left, top, right, bottom], outline="red", width=3)

    # Visualize faces with bounding boxes for Selfie image
    selfie_image_pil = Image.fromarray(selfie_image)  # Convert to PIL Image
    draw_selfie = ImageDraw.Draw(selfie_image_pil)

    for (top, right, bottom, left) in selfie_face_locations:
        draw_selfie.rectangle([left, top, right, bottom], outline="blue", width=3)

    # Display images with bounding boxes
    plt.figure(figsize=(12, 6))

    # CNIC Image with Bounding Boxes
    plt.subplot(1, 2, 1)
    plt.imshow(cnic_image_pil)
    plt.title("CNIC Image with Faces")
    plt.axis('off')

    # Selfie Image with Bounding Boxes
    plt.subplot(1, 2, 2)
    plt.imshow(selfie_image_pil)
    plt.title("Selfie Image with Faces")
    plt.axis('off')

    # Show both images
    plt.show()

# Define file paths for CNIC and selfie image
cnic_image_path = "cnic9.png"  # Path to the CNIC image

# Step 1: Perform liveness check
print("Performing liveness check...")
if liveness_check():
    # Step 2: Capture the selfie after liveness is confirmed
    selfie_image_path = capture_selfie()  # Capture the selfie dynamically

    # Step 3: Compare the CNIC and selfie images if selfie is captured
    if selfie_image_path:
        compare_faces(cnic_image_path, selfie_image_path)
else:
    print("Liveness check failed. Please try again.")
