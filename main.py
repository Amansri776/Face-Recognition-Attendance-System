import cv2
import numpy as np
import face_recognition

# Load Images
imgModi = face_recognition.load_image_file(r"C:\code\project\Images_Attendance\MODI JI.jpg")
imgModi = cv2.cvtColor(imgModi, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file(r"C:\code\project\Images_Attendance\Narendra_Modi.webp")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Detect Faces and Encode
faceloc = face_recognition.face_locations(imgModi)[0]
encodeModi = face_recognition.face_encodings(imgModi)[0]
cv2.rectangle(imgModi, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]  # Fixed index error
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (155, 0, 255), 2)

# Compare Faces
results = face_recognition.compare_faces([encodeModi], encodeTest)
faceDis = face_recognition.face_distance([encodeModi], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

# Show Images
cv2.imshow('modi', imgModi)
cv2.imshow('narendra-modi', imgTest)
cv2.waitKey(0)  # Fixed function name
cv2.destroyAllWindows()
