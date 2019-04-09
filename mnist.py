import yaml
from network import Network
import numpy as np

config = yaml.load(open("config.yaml"))

layer_sizes = config["layer_sizes"]
epochs = config["epochs"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
input_divider = config["input_divider"]

images = open("training_data/train_images.idx3-ubyte", "rb")
labels = open("training_data/train_labels.idx1-ubyte", "rb")

images.read(4)
number_of_images = int.from_bytes(images.read(4), byteorder="big")
width = int.from_bytes(images.read(4), byteorder="big")
height = int.from_bytes(images.read(4), byteorder="big")
image_size = width * height
labels.read(8)

def get_expected_output(x):
        return np.array(x * [ 0 ] + [ 1 ] + (10 - x - 1) * [ 0 ], dtype=float)

training_data = [ ]

for i in range(0, number_of_images):
        image = np.fromfile(images, dtype=np.uint8, count=image_size) / input_divider
        label = int.from_bytes(labels.read(1), byteorder="big") 
        training_data.append((image, get_expected_output(label)))

network = Network(training_data, layer_sizes, epochs, batch_size, learning_rate)
network.load() # load network if exists
network.train()
network.save()