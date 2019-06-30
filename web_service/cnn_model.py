import numpy as np
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image, ImageOps


class Model(object):
    def __init__(self, train_new_model=False):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        if train_new_model:
            self.model = self._new_model(128, 10)
            self.compute_accuracy()
        else:
            with open('saved_weights/saved_model.json', 'r') as f:
                self.model = model_from_json(f.read())
            self.model.load_weights('saved_weights/model_weights.h5')

    @staticmethod
    def _load_dataset():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test

    def _pre_processing_data(self):
        img_rows, img_cols = 28, 28

        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_rows, img_cols, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        num_classes = len(np.unique(self.y_train))
        self.y_train = to_categorical(self.y_train, num_classes)
        self.y_test = to_categorical(self.y_test, num_classes)

        return input_shape, num_classes

    @staticmethod
    def _initialize_model(input_shape, num_classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model

    def _new_model(self, batch_size, num_epoch):
        self.x_train, self.y_train, self.x_test, self.y_test = self._load_dataset()
        input_shape, num_classes = self._pre_processing_data()

        new_model = self._initialize_model(input_shape, num_classes)
        new_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
        new_model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=num_epoch, verbose=1, validation_data=(self.x_test, self.y_test))

        return new_model

    def compute_accuracy(self):
        if self.x_train is None or self.y_train is None or self.x_test is None or self.y_test is None:
            self.x_train, self.y_train, self.x_test, self.y_test = self._load_dataset()
            _, _ = self._pre_processing_data()

            self.model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print("using the MNIST Dataset")
        print("Train loss:", score[0])
        print("Train accuracy:", score[1])
        print("-----------------------")
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    def inference(self):
        img = Image.open("images/savedImage.png")
        img.load()
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        gray_image = background.convert('L')
        gray_image = ImageOps.invert(gray_image)
        gray_image = gray_image.resize((28, 28), Image.ANTIALIAS)
        gray_image = np.asarray(gray_image, dtype='float32')
        gray_image = np.expand_dims(gray_image, axis=0)
        gray_image = np.expand_dims(gray_image, axis=-1)
        gray_image = gray_image / 255

        predicted_value = np.argmax(self.model.predict(gray_image))
        K.clear_session()

        return predicted_value


def main():
    model = Model()
    print(model.inference())


if __name__ == '__main__':
    main()
