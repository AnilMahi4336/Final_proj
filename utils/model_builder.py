import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")
from models.Model import build_model

def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])


def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="ResNet101", is_training=True):
	print("Preparing the model ...")

	if "ResNet50" == frontend and not os.path.isfile("models/resnet_v2_50.ckpt"):
	    download_checkpoints("ResNet50")
	if "ResNet101" == frontend and not os.path.isfile("models/resnet_v2_101.ckpt"):
	    download_checkpoints("ResNet101")

	network = None
	init_fn = None
	
	if model_name == "Model":
	    network = build_model(net_input, num_classes)
	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	return network, init_fn
