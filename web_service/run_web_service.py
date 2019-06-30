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
        img_data = img.encode('utf-8')
        
        with open("images/savedImage.png", "wb") as fh:
            fh.write(base64.b64decode(img_data))

        model = Model()
        num = model.inference()

        return {'img': int(num)}


api.add_resource(Communicator, '/<string:img>')


if __name__ == '__main__':
    '''
    Se trocar o IP, lembra de trocar tbm na linha 50 do drawing.js
    '''
    app.run('192.168.0.100', debug=False)

