import cv2
from keras_facenet import FaceNet
from mtcnn import MTCNN

embedder = FaceNet()
detector = MTCNN()

def extract_face_embedding(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]["box"]

    x = max(0, x)
    y = max(0, y)

    face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (160,160))

    embedding = embedder.embeddings([face])[0]

    return embedding.reshape(1,-1)
