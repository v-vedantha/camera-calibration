import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_iris = mp.solutions.iris

# For static images:
IMAGE_FILES = ['calibration_image_0.jpg']
#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
#with mp_face_mesh.FaceMesh(
#    static_image_mode=True,
#    max_num_faces=1,
#    min_detection_confidence=0.5) as face_mesh:
#  for idx, file in enumerate(IMAGE_FILES):
#    image = cv2.imread(file)
#    # Convert the BGR image to RGB before processing.
#    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#    breakpoint()
#
#    # Print and draw face mesh landmarks on the image.
#    if not results.multi_face_landmarks:
#      continue
#    annotated_image = image.copy()
#    for face_landmarks in results.multi_face_landmarks:
#      print('face_landmarks:', face_landmarks)
#      mp_drawing.draw_landmarks(
#          image=annotated_image,
#          landmark_list=face_landmarks,
#          connections=mp_face_mesh.FACE_CONNECTIONS,
#          landmark_drawing_spec=drawing_spec,
#          connection_drawing_spec=drawing_spec)
#    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
#
#mp_face_detection = mp.solutions.face_detection
#mp_drawing = mp.solutions.drawing_utils
#
## For static images:
#with mp_face_detection.FaceDetection(
#    model_selection=1, min_detection_confidence=0.5) as face_detection:
#  for idx, file in enumerate(IMAGE_FILES):
#    image = cv2.imread(file)
#    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
#    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#    breakpoint()
#    # Draw face detections of each face.
#    if not results.detections:
#      continue
#    annotated_image = image.copy()
#    for detection in results.detections:
#      print('Nose tip:')
#      print(mp_face_detection.get_key_point(
#          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
#      mp_drawing.draw_detection(annotated_image, detection)
#    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

mp_face_detection = mp.solutions.face_detection

# For static images:
with mp_iris.Iris() as iris:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = iris.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #breakpoint()
    # Draw face detections of each face.
    if not results.face_landmarks_with_iris:
      continue
    annotated_image = image.copy()
    for face_landmark_with_iris in [results.face_landmarks_with_iris]:
      print('face_landmarks:', face_landmark_with_iris)
      mp_drawing.draw_iris_landmarks(annotated_image,results.face_landmarks_with_iris)
      mp_drawing.draw_landmarks(annotated_image,results.face_landmarks_with_iris)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
  