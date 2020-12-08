import tensorflow as tf
import cv2
import numpy as np
import matplotlib.cm as cm
import os
import matplotlib.pyplot as plt

class GradCAM(object):

    def __init__(self,image:str,model,last_conv_layer_name:str,classifier_layer_names:list,IMAGE_SIZE:int):
        self.model=model
        self.image=image
        self.last_conv_layer_name=last_conv_layer_name
        self.classifier_layer_names=classifier_layer_names
        self.IMAGE_SIZE=IMAGE_SIZE

    def make_gradcam_heatmap(
        self,img_array, model, last_conv_layer_name, classifier_layer_names
    ):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in classifier_layer_names:
            x = model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            # Compute class predictions
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        return heatmap


    def create_superimposed_visualization(self,img, heatmap):

        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))

        # heatmap = np.uint8(255*heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img

        return superimposed_img

    def plot_cam(self):    
        last_conv_layer_name=self.last_conv_layer_name
        # We also need the names of all the layers that are part of the model head
        classifier_layer_names = self.classifier_layer_names

        img=cv2.imread(self.image)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(self.IMAGE_SIZE,self.IMAGE_SIZE))

        img=img/255.

        fig, ax = plt.subplots(figsize=(15, 10), ncols=2, nrows=1)

        raw_image = img

        image = np.expand_dims(raw_image, axis=0)

        heatmap = self.make_gradcam_heatmap(
            image, model, last_conv_layer_name, classifier_layer_names)
        superimposed_image = self.create_superimposed_visualization(raw_image, heatmap)

        ax[0].imshow(raw_image)
        ax[1].imshow(superimposed_image)
        plt.title("Grad CAM Visualization")
        plt.show()
