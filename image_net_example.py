#TODO remove this file, it's just to serve as a reference for creating our own SHAP pipeline
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import shap
import pdb
import matplotlib.pyplot as plt

# load pre-trained model and data
model = ResNet50(weights='imagenet')
X, y = shap.datasets.imagenet50()

# getting ImageNet 1000 class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]
#print("Number of ImageNet classes:", len(class_names))
#print("Class names:", class_names)


# function to get model output; replace this function with your own model function.
def f(x):
    tmp = x.copy()
    preprocess_input(tmp)
    return model(tmp)

# define a masker that is used to mask out partitions of the input image.
masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)

# create an explainer with model and image masker
explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names, algorithm="partition")

# here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
input_images = X[1:3]

# show the top 4 predicted classes and their explanations
outputs = shap.Explanation.argsort.flip[:4]
shap_values = explainer_blur(input_images, max_evals=5000, batch_size=50, outputs=outputs)

pdb.set_trace()

# output with shap values
shap.image_plot(shap_values)