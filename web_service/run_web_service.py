import base64
from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api
from cnn_model import Model

app = Flask(__name__)
CORS(app)
api = Api(app)


class Communicator(Resource):
    @staticmethod
    def get(img):
        print(img)
        img = img.replace("|", "/")        
        # img_data = bytes(img, 'utf-8')
        img_data = img.encode('utf-8')
        
        with open("imageToSave.png", "wb") as fh:
            #fh.write(base64.decodebytes(img_data))
            fh.write(base64.b64decode(img_data))

        model = Model()
        num = model.inference()
        return {'img': num}


api.add_resource(Communicator, '/<string:img>')


if __name__ == '__main__':
    # app.run(debug=True)
    '''
    Se trocar o IP, lembra de trocar tbm na linha 90 do drawing.js
    '''
    app.run('192.168.1.60',debug=True)

