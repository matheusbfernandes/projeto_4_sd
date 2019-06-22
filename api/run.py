from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api
import base64

app = Flask(__name__)
CORS(app)
api = Api(app)

class HelloWorld(Resource):
    def get(self, img):
        # Transformando em imagem e salvando no pc
        img = img.replace("|", "/")
        img_data = bytes(img, 'utf-8')
        with open("imageToSave.png", "wb") as fh:
            fh.write(base64.decodebytes(img_data))
        return {'img': 'ok'}


api.add_resource(HelloWorld, '/<string:img>')

if __name__ == '__main__':
    app.run(debug=True)