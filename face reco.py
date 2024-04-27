import cv2 
from simple_facerec import SimpleFacerec

# Initialize SimpleFacerec and load encoding images
sfr = SimpleFacerec()
sfr.load_encoding_images("imgs/")

# Load camera
cap = cv2.VideoCapture(0)

# Initialize variables for accuracy calculation
total_faces = 0
correctly_recognized = 0

while True:
    ret, frame = cap.read()

    # Detect known faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Display name and draw rectangle around the face
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

        # Increment total faces
        total_faces += 1

        # Check if the recognized name matches the ground truth
        # Replace 'ground_truth_name' with the actual ground truth name for this face
        if name == 'veena':
            correctly_recognized += 1

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Esc key
        break

# Calculate accuracy
accuracy = (correctly_recognized / total_faces) * 100
print("Accuracy: {:.2f}%".format(accuracy))

cap.release()
cv2.destroyAllWindows()
