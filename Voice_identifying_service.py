# IMPORT NECESSARY MODULES
import tensorflow as tf
import numpy as np
import librosa

# FILE PATHS
MODEL_PATH = "voice_model.h5"

# SET NECESSARY PARAMETERS
NUMBER_SAMPLES_TO_CONSIDER = 66150

# IDENTIFYING SERVICE CLASS
class _voice_identifying_service:

  model = None
  _mappings = [
     "0",
     "101",
     "15",
     "222",
     "3",
     "30",
     "54",
     "635",
     "7",
     "747",
     "88",
     "9"
  ]
  _instance = None

  # FUNCTION FOR PREPROCESS INPUT AUDIO
  def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    # LOAD AUDIO FILE
    signal, sr = librosa.load(file_path)

    # ENSURE MINIMUM 3 SECs
    if len(signal) > NUMBER_SAMPLES_TO_CONSIDER:
      signal =  signal[:NUMBER_SAMPLES_TO_CONSIDER]

    # EXTARCT MFCCs
    MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    return MFCCs.T

  # FUNCTION FOR IDENTIFY AUDIO
  def predict(self, file_path):
    # EXTRACT MFCC
    MFCCs = self.preprocess(file_path)

    # CONVERT 2D MFCCs ARRAY INTO 4D ARRAY
    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

    # MAKE PREDICTION
    predictions = self.model.predict(MFCCs)
    predicted_index = np.argmax(predictions)
    predicted_keyword = self._mappings[predicted_index]

    return predicted_keyword

# FUNCTION FOR ENSURE ONLY ONE INSTENCE OF IDENTIFYING SERVICE CLASS
def voice_identifying_service():
  if _voice_identifying_service._instance is None:
    _voice_identifying_service._instance = _voice_identifying_service()
    _voice_identifying_service.model = tf.keras.models.load_model(MODEL_PATH)

  return _voice_identifying_service._instance

if __name__ == "__main__":
  identify_voice = voice_identifying_service()

  audio = "635.wav"

  prediction = identify_voice.predict(audio)

  print(f"Predicted keyword: {prediction}")