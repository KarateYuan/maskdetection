import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import model_from_json

def load_keras_model(json_path, weight_path):
    model = model_from_json(open('models/face_mask_detection.json').read())
    model.load_weights('models/face_mask_detection.hdf5')
    return model


def convert2tflite(model):
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("face_keras_convert.tflite", "wb").write(tflite_model)

if __name__ == "__main__":
    json_path = 'models/face_mask_detection.json'
    weight_path = 'models/face_mask_detection.hdf5'
    model = load_keras_model(json_path, weight_path)
    convert2tflite(model)
    