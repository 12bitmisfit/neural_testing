import os
# Disables GPU to run on CPU, comment out if you want it to run on GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential, model_from_json
from keras.layers import Dense, InputLayer, Flatten, Reshape, BatchNormalization, Dropout, Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import numpy
from PIL import Image
import math
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

train_dir = "C:/videos/clean/120/"
validation_dir = "C:/videos/clean/240/"
batch_size = 50
epochs = 50
load = False
dh = 426
dw = 240


def resize(image):
    image = tf.image.resize(image, size=(240, 426), method="nearest")


train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1/255.0
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(240, 426),
    batch_size=batch_size,
    class_mode="input",
    color_mode='rgb',
)


validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(240, 426),
    batch_size=batch_size,
    class_mode="input",
    color_mode='rgb'
)



def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

# define the keras model
model = Sequential([
    InputLayer(input_shape=(240, 426, 3)),
    Conv2D(64, (3, 3), data_format='channels_last', padding="same", activation='relu'),
    Conv2D(64, (3, 3), data_format='channels_last', padding="same", activation='relu'),
    Conv2D(64, (3, 3), data_format='channels_last', padding="same", activation='relu'),
    Conv2D(64, (3, 3), data_format='channels_last', padding="same", activation='relu'),
    Conv2D(3, (9, 9), data_format='channels_last', padding="same", activation='relu')
])
# loads previously trained model if it exists
if os.path.exists('model.json') == True and load == True:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

# compile the keras model
#adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mae', optimizer="Adam", metrics=PSNRLoss)
# fit the keras model on the dataset
model.fit(train_generator, validation_data=validation_generator, epochs=epochs, batch_size=batch_size)
print(model.summary())

# Save the model and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


output_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(240, 426),
    batch_size=1,
    class_mode="input",
    color_mode='rgb',
)

test1, test2 = output_generator.next()

prediction = model.predict(test1)
prediction = numpy.squeeze(prediction, axis=0)
prediction = prediction * 255
prediction = numpy.uint8(prediction)
print(prediction.shape)
img = Image.fromarray(prediction, 'RGB')
img.show()
