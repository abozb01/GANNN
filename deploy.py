# SmartPhone
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('trained_model.h5')

# Converting model to format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Saving converted model 
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


# IoT Device
import tensorflow as tf
from tensorflow import lite as tflite

# Load trained model
model = tf.keras.models.load_model('trained_model.h5')

# Convert the model to TensorFlow Lite for Microcontrollers format
converter = tflite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  
tflite_model = converter.convert()

# Saving converted model 
with open('model_microcontroller.tflite', 'wb') as f:
    f.write(tflite_model)

