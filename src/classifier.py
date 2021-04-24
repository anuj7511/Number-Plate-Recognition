import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    MaxPooling2D,
    Dropout,
    Conv2D,
    BatchNormalization,
)
import tensorflow.keras.backend as K


def get_model(num_classes):
    K.clear_session()

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), activation="relu", strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), strides=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizers.Adam(lr=0.0001),
        metrics=["accuracy"],
    )

    return model


class ReadPlate:
    def __init__(
        self,
        digits_path="models/digits-v6-20.h5",
        alphabets_path="models/alphabets-v6.h5",
        all36_path="models/all36-v6-20.h5",
    ):
        self.digits_model = get_model(10)
        self.alphabets_model = get_model(26)
        self.all36_model = get_model(36)

        self.digits_model.load_weights(digits_path)
        self.alphabets_model.load_weights(alphabets_path)
        self.all36_model.load_weights(all36_path)

        self.labels36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.labels_alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.labels_digits = "0123456789"
        self.state_codes = [
            "AN",
            "AP",
            "AR",
            "AS",
            "BR",
            "CH",
            "PB",
            "CG",
            "DD",
            "DL",
            "GA",
            "GJ",
            "HR",
            "HP",
            "JK",
            "JH",
            "KL",
            "LA",
            "LD",
            "MP",
            "MH",
            "MN",
            "ML",
            "MZ",
            "NL",
            "OD",
            "PY",
            "RJ",
            "SK",
            "TN",
            "TS",
            "TR",
            "UP",
            "UK",
            "WB",
        ]

    def predict_char(self, char_list):
        fixed_imgs = [self.fix_dimension(char) for char in char_list]

        # (9,10,11) are the only possible correct length of license plate numbers
        if 9 <= len(fixed_imgs) <= 11:
            state_code = [
                self.labels_alphabets[self.alphabets_model.predict(img).argmax(-1)[0]]
                for img in fixed_imgs[:2]
            ]
            last_four = [
                self.labels_digits[self.digits_model.predict(img).argmax(-1)[0]]
                for img in fixed_imgs[-4:]
            ]
            remaining_images = fixed_imgs[2:-4]
            remaining_preds = []

            # Only Delhi registered cars can have a 1 digit district code, having a special code for it allows
            if ("".join(state_code) == "DL") or (
                "".join(state_code) not in self.state_codes
            ):
                for img in remaining_images[:1]:
                    remaining_preds.append(
                        self.labels_digits[self.digits_model.predict(img).argmax(-1)[0]]
                    )
                for img in remaining_images[1:-1]:
                    remaining_preds.append(
                        self.labels36[self.all36_model.predict(img).argmax(-1)[0]]
                    )
                for img in remaining_images[-1:]:
                    remaining_preds.append(
                        self.labels_alphabets[
                            self.alphabets_model.predict(img).argmax(-1)[0]
                        ]
                    )
            else:
                for img in remaining_images[:2]:
                    remaining_preds.append(
                        self.labels_digits[self.digits_model.predict(img).argmax(-1)[0]]
                    )
                for img in remaining_images[2:]:
                    remaining_preds.append(
                        self.labels_alphabets[
                            self.alphabets_model.predict(img).argmax(-1)[0]
                        ]
                    )

            preds = state_code + remaining_preds + last_four

        # since we dont have the correct number of identified characters, ie some were missed or additional contours were identified too
        # thus no specific structure, so we treat all as a general case
        else:
            preds = []
            for img in fixed_imgs:
                preds.append(self.labels36[self.all36_model.predict(img).argmax(-1)[0]])

        fixed_imgs = [img[0, :, :, 0] for img in fixed_imgs]
        return fixed_imgs, preds

    @staticmethod
    def fix_dimension(img):
        """Dummy Batch image arrays"""
        assert img.shape == (
            28,
            28,
        ), f"Input Dimensions expected (28,28) got {img.shape}"
        new_img = np.zeros((28, 28, 3))
        for i in range(3):
            new_img[:, :, i] = img
        new_img = new_img.reshape(1, 28, 28, 3)
        return new_img / 255.0


if __name__ == "__main__":

    readPlate = ReadPlate(
        digits_path="models/digits-v6-20.h5",
        alphabets_path="models/alphabets-v6.h5",
        all36_path="models/all36-v6-20.h5",
    )
