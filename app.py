import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from io import StringIO
from transformers import pipeline

# Streamlit application title
st.title("CNN Layer Visualization and Explanation")

# Sidebar for user input
st.sidebar.header("Settings")

# Add more prebuilt models to the list of options
model_choice = st.sidebar.selectbox(
    "Choose a CNN model:",
    ["VGG16", "ResNet50", "InceptionV3", "MobileNetV2", "Xception", "EfficientNetB0"]
)

# File uploader to allow the user to upload an image
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Filter selection for displaying the number of feature maps (restrict to 1-6 filters)
num_filters = st.sidebar.slider("Number of filters to display per layer:", 1, 6, 6)

# Function to create and return a CNN model
def get_model(choice):
    if choice == "VGG16":
        return tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    elif choice == "ResNet50":
        return tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
    elif choice == "InceptionV3":
        return tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
    elif choice == "MobileNetV2":
        return tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
    elif choice == "Xception":
        return tf.keras.applications.Xception(weights='imagenet', include_top=False)
    elif choice == "EfficientNetB0":
        return tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)
    else:
        # Example of a custom CNN model
        model = models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        return model

# Function to preprocess the image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Preprocess for VGG16
    return img, img_array

# Function to visualize feature maps
def plot_layer_activation(activation, layer_name, num_filters=6):
    """
    Plots the feature maps of a single convolutional layer.
    :param activation: Activation output of a specific layer.
    :param layer_name: Name of the convolutional layer.
    :param num_filters: Number of feature maps to display (set to 6 by default).
    """
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 20))
    fig.suptitle(f"Layer: {layer_name}", fontsize=16)
    for i in range(num_filters):
        filter_img = activation[0, :, :, i]  # Extract the i-th feature map
        axes[i].imshow(filter_img, cmap='viridis')
        axes[i].axis('off')
    st.pyplot(fig)

# Simulate LLM explanation for convolutional layer output
def explain_layer_output(layer_name):
    """
    Provides a simple explanation for what each convolutional layer is likely detecting.
    :param layer_name: Name of the convolutional layer.
    :return: A textual explanation of the layer's function.
    """
    if "conv1" in layer_name or "block1" in layer_name:
        return "This layer detects simple patterns such as edges and textures in the image. It focuses on low-level features like lines and gradients."
    elif "conv2" in layer_name or "block2" in layer_name:
        return "This layer builds upon lower-level features and starts detecting more complex shapes, such as corners and textures."
    elif "conv3" in layer_name or "block3" in layer_name:
        return "This layer identifies more complex objects, such as combinations of shapes. It's more abstract and builds up details from previous layers."
    else:
        return "This layer captures high-level, abstract representations of the image, focusing on detailed object patterns or specific regions of interest."

# Function to summarize text using a BART model for summarization
def summarize_text(prompt):
   
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(prompt, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Only run the rest of the code if an image has been uploaded
if uploaded_file is not None:
    # Left and Right columns layout with more space for the right side
    col1, col2 = st.columns([1, 2])  # The right column is twice as wide as the left column

    # Display the original uploaded image in the left column
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Load and preprocess the image
    img, img_array = preprocess_image(uploaded_file)

    # Load the selected model
    model = get_model(model_choice)

    # Display the model architecture summary below the layer visualization
    st.subheader("Model Architecture")
    with StringIO() as buffer:
        model.summary(print_fn=lambda x: buffer.write(x + '\n'))
        st.text(buffer.getvalue())

    # Display model summary using BART model
    st.subheader("Model Summary")
    if(model_choice != 'Custom CNN'):

        gpt_summary = summarize_text(f"Please provide a brief summary of the {model_choice} deep learning model.")
        st.write("GPT Model Summary:")
        st.write(gpt_summary)
    else:
        st.write('It is a Custom Model')

    # Create a model that will output the activations of all convolutional layers
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'block' in layer.name]  # Get the convolutional layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # Get the activations for the input image
    activations = activation_model.predict(img_array)

    # Get the names of all convolutional layers
    layer_names = [layer.name for layer in model.layers if 'conv' in layer.name or 'block' in layer.name]

    # Let the user choose a specific layer number
    selected_layer = st.sidebar.selectbox(f"Choose a layer (1-{len(layer_names)})", list(range(1, len(layer_names) + 1)))

    # Display the activations of the selected layer in the right column
    with col2:
        plot_layer_activation(activations[selected_layer - 1], layer_names[selected_layer - 1], num_filters=num_filters)

        # LLM-like explanation of what the layer is doing
        st.subheader("Explanation of Layer Output")
        layer_explanation = explain_layer_output(layer_names[selected_layer - 1])
        st.write(layer_explanation)
