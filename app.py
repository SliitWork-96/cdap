from flask import Flask, jsonify, request
import numpy as np
import pickle
import flask
from flask import Flask, request
import werkzeug
import cv2
import numpy as np
import tensorflow as tf

from Voice_identifying_service import voice_identifying_service
from LetterDysgraphiaController import letterPrediction

app = Flask(__name__)

sinhalaLetterPrediction = letterPrediction()

@app.route('/test', methods = ['GET', 'POST'])
def predictNew():
  greet = "26"
  return jsonify({ 'msg': "Hello, " + greet})

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    # SAVE AUDIO FILE
    audiofile = flask.request.files['file']
    filename = werkzeug.utils.secure_filename(audiofile.filename)
    audiofile.save('Audio/' + filename)

    # INVOKE SPOTTING SYSTEM
    identify_voice = voice_identifying_service()

    # # MAKE PREDICTION
    predicted_number = identify_voice.predict('Audio/' + filename)

    # REMOVE AUDIO FILE
    # # os.remove(fileName)

    # SEND BACK PREDICTION
    voice_dataset = {'Keyword': predicted_number}
    print(predicted_number)

    return predicted_number

@app.route("/letter",methods=['GET', 'POST'])
def scrrenLetterDysgraphia():
    model = tf.keras.models.load_model('trained_model3.h5')
    input_code = request.form['input_code']
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('Upload/' + filename)
    img = cv2.imread('Upload/' + filename)
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    img = img.reshape(1, 32, 32, 1)
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)

    print(input_code)
    print(classIndex)

    input_code = int(input_code)
    if classIndex == input_code:
        return str(probVal)
    else:
        return str(probVal * -1)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)