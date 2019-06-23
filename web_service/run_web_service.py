import base64
from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api


app = Flask(__name__)
CORS(app)
api = Api(app)


class Communicator(Resource):
    @staticmethod
    def get(img):
        img = img.replace("|", "/")
        img_data = bytes(img, 'utf-8')
        with open("imageToSave.png", "wb") as fh:
            fh.write(base64.decodebytes(img_data))
        return {'img': 'ok'}


api.add_resource(Communicator, '/<string:img>')


if __name__ == '__main__':
    app.run(debug=True)
