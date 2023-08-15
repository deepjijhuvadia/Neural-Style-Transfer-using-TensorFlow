# import numpy, tensorflow and matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import VGG 19 model and keras Model API
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import load_img, img_to_array
from keras.models import Model

# Image Credits: Tensorflow Doc
content_path = tf.keras.utils.get_file('content.jpg',
                                       'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('style.jpg',
                                     'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

# code to load and process image
def load_and_process_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# code to deprocess image
def deprocess(img):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# code to display image
def display_image(image):
    if len(image.shape) == 4:
        img = np.squeeze(image, axis=0)
        img = deprocess(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    return

# load content image
content_img = load_and_process_image(content_path)
display_image(content_img)

# load style image
style_img = load_and_process_image(style_path)
display_image(style_img)

# define content model
model = VGG19(include_top=False, weights='imagenet')
model.trainable = False
content_layer = 'block5_conv2'
content_model = Model(inputs=model.input, outputs=model.get_layer(content_layer).output)
content_model.summary()

# define style model
style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']
style_models = [Model(inputs=model.input, outputs=model.get_layer(layer).output) for layer in style_layers]

# Content loss
def content_loss(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    loss = tf.reduce_mean(tf.square(a_C - a_G))
    return loss

# Gram matrix
def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

weight_of_layer = 1. / len(style_models)

# Style loss
def style_loss(style, generated):
    J_style = 0
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += current_cost * weight_of_layer
    return J_style

# Total variation loss
def total_variation_loss(image):
    x_var = tf.reduce_mean(tf.square(image[:, :, 1:, :] - image[:, :, :-1, :]))
    y_var = tf.reduce_mean(tf.square(image[:, 1:, :, :] - image[:, :-1, :, :]))
    return x_var + y_var

# Training function
generated_images = []

def training_loop(content_path, style_path, iterations=50, a=10, b=1000, lr=7.0):
    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    generated = tf.Variable(content, dtype=tf.float32)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    best_cost = float('inf')
    best_image = None

    for i in range(iterations):
        with tf.GradientTape() as tape:
            J_content = content_loss(content, generated)
            J_style = style_loss(style, generated)
            J_tv = total_variation_loss(generated)
            J_total = a * J_content + b * J_style + J_tv

        grads = tape.gradient(J_total, generated)
        opt.apply_gradients([(grads, generated)])

        if J_total < best_cost:
            best_cost = J_total
            best_image = generated.numpy()

        print("Iteration: {}".format(i))
        print('Total Loss: {:.4e}.'.format(J_total))

        generated_images.append(generated.numpy())

    return best_image

# Train the model and get the best image
final_img = training_loop(content_path, style_path)

# Code to display the best generated image and last 10 intermediate results
plt.figure(figsize=(12, 12))
for i in range(10):
    plt.subplot(4, 3, i + 1)
    display_image(generated_images[i + 39])
plt.show()

# Plot the best result
display_image(final_img)