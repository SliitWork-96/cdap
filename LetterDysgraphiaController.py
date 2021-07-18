import flask
import werkzeug
import Test


class letterPrediction:
        def handle_request():
            t = Test
            imagefile = flask.request.files['image']
            filename = werkzeug.utils.secure_filename(imagefile.filename)
            # print("\nReceived image File name : " + imagefile.filename)
            imagefile.save('Upload/'+filename)
            # filename = 'test.jpg'
            result = t.test(filename)
            # print(result)

            return str(result)

            # return str(result)