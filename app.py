import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
from utils.face_utils import extract_face_embedding
from utils.voice_utils import extract_voice_features
from utils.prediction import make_prediction

import warnings
warnings.filterwarnings("ignore")

FACE_THRESHOLD = 0.70
VOICE_THRESHOLD = 0.55

# Load Models
print("Loading models...")
face_model = joblib.load("models/facial_recognition_rf_model.pkl")
voice_model = joblib.load("models/voiceprint_verification_model.pkl")
product_model = joblib.load("models/product_recommendation_model.pkl")
print("Models loaded successfully.\n")


# FACE AUTHENTICATION
print("=" * 55)
print("STEP 1: FACE RECOGNITION")
print("=" * 55)

image_path = input("Enter face image path: ").strip()
embedding = extract_face_embedding(image_path)

if embedding is None:
    print("\nNo face detected.")
    print("ACCESS DENIED.")
    exit()

# Predict probabilities
face_probs = face_model.predict_proba(embedding)[0]
face_confidence = face_probs.max()
person = face_model.classes_[face_probs.argmax()]

if face_confidence < FACE_THRESHOLD:
    print("\nFace not recognized.")
    print(f"Confidence: {face_confidence:.2%}")
    print("ACCESS DENIED.")
    exit()

print(f"\nFace recognized as: {person}")
#print(f"Confidence: {face_confidence:.2%}")


# VOICE AUTHENTICATION
print("\n" + "=" * 55)
print("STEP 2: VOICE VERIFICATION")
print("=" * 55)

print("-- Note the provided voice record must be in WAV format (.wav) --")
audio_path = input("Enter voice recording path: ").strip()
voice_features = extract_voice_features(audio_path)

if not audio_path.lower().endswith(".wav"):
    print("Please provide a WAV recording.")
    exit()

voice_probs = voice_model.predict_proba(voice_features)[0]
voice_confidence = voice_probs.max()

speaker = voice_model.classes_[voice_probs.argmax()]

if voice_confidence < VOICE_THRESHOLD:
    print("\nVoice not recognized.")
    print(f"Confidence: {voice_confidence:.2%}")
    print("ACCESS DENIED.")
    exit()

print(f"\nVoice recognized as: {speaker}")
#print(f"Confidence: {voice_confidence:.2%}")


# IDENTITY MATCH
if speaker != person:
    print("\nAuthentication Failed!")
    print("Face and voice belong to different users.")
    print("ACCESS DENIED!")
    exit()

print("\nIdentity verified successfully!")


# PRODUCT RECOMMENDATION
print("\n" + "=" * 55)
print("STEP 3: PRODUCT PREDICTION - PRODUCT RECOMMENDATION SYSTEM")
print("=" * 55)

make_prediction(product_model)

print("\n" + "=" * 55)
print("SESSION COMPLETED SUCCESSFULLY")
print("=" * 55)