import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model

def load_keras_model(json_path, weight_path):
    model = model_from_json(open('models/face_mask_detection.json').read())
    model.load_weights('models/face_mask_detection.hdf5')
    return model


def keras_structure(model):
    model.summary()

def print_structure(model):
    plot_model(model,to_file='model_keras.png',show_shapes=True)



if __name__ == "__main__":
    model = load_keras_model('models/face_mask_detection.json', 'models/face_mask_detection.hdf5')
    keras_structure(model)
    #print_structure(model)
    