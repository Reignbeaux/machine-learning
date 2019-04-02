import yaml
from network import Network

config = yaml.load(open("config.yaml"))
inner_layers = config["inner_layers"]

network = Network()
