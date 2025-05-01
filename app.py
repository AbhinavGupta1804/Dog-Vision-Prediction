import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import numpy as np
from PIL import Image
import os

# Define the FeatureExtractorLayer
class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, feature_extractor_url=None, trainable=False, **kwargs):
        super(FeatureExtractorLayer, self).__init__(**kwargs)
        self.feature_extractor_url = feature_extractor_url
        self.trainable_setting = trainable
        
        if feature_extractor_url:
            self.feature_extractor = hub.KerasLayer(
                feature_extractor_url, 
                trainable=trainable
            )
        else:
            self.feature_extractor = None

    def build(self, input_shape):
        if self.feature_extractor is None and self.feature_extractor_url:
            self.feature_extractor = hub.KerasLayer(
                self.feature_extractor_url,
                trainable=self.trainable_setting
            )
        super().build(input_shape)

    def call(self, inputs):
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized")
        return self.feature_extractor(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_extractor_url': self.feature_extractor_url,
            'trainable': self.trainable_setting
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        url = config.pop('feature_extractor_url', None)
        trainable = config.pop('trainable', False)
        return cls(feature_extractor_url=url, trainable=trainable, **config)

# Load model
@st.cache_resource
def load_model():
    custom_objects = {
        "FeatureExtractorLayer": FeatureExtractorLayer
    }
    model = tf.keras.models.load_model("best_model_tf.h5", custom_objects=custom_objects)
    return model

# Load breed names
@st.cache_data
def load_breeds():
    import pandas as pd
    df = pd.read_csv("labels.csv")
    breeds = sorted(df["breed"].unique())
    return breeds

# Preprocess image
def preprocess_image(image):
    IMG_SIZE = 224
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Streamlit app UI
st.title("Dog Vision Prediction")
st.write("""
Upload a photo of a dog, and this app will predict the breed using a deep learning model trained on MobileNetV2.
""")

# Sidebar Info
with st.sidebar:
    st.header("About")
    st.write("""
    This web app uses a deep learning model to identify dog breeds from uploaded images.

    It supports the classification of dozens of popular breeds, such as:
    - Labrador Retriever
    - German Shepherd
    - Bulldog
    - Golden Retriever
    - Poodle
    - Beagle
    - Rottweiler
    - Boxer
    - Yorkshire Terrier
    - ...and many more.
    """)

    st.header("How it works")
    st.write("""
    1. Upload an image of a dog.
    2. The model analyzes the image using a trained CNN.
    3. You’ll get the **top 3 predicted breeds** with their confidence scores.
    """)
# Load model and breed list
model = load_model()
breeds = load_breeds()

# Upload image
uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=False, width=300)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    top_indices = np.argsort(prediction[0])[-3:][::-1]

    st.write("## Predictions:")
    for i, idx in enumerate(top_indices):
        breed_name = breeds[idx]
        confidence = prediction[0][idx] * 100
        st.write(f"**{i+1}. {breed_name}** — Confidence: {confidence:.2f}%")


# import tensorflow as tf
# import tensorflow_hub as hub
# import streamlit as st
# import numpy as np
# from PIL import Image
# import io
# import requests
# from io import BytesIO
# import os

# # Define the same FeatureExtractorLayer class
# class FeatureExtractorLayer(tf.keras.layers.Layer):
#     def __init__(self, feature_extractor_url=None, trainable=False, **kwargs):
#         super(FeatureExtractorLayer, self).__init__(**kwargs)
#         self.feature_extractor_url = feature_extractor_url
#         self.trainable_setting = trainable
        
#         # Create feature extractor if URL is provided
#         if feature_extractor_url:
#             self.feature_extractor = hub.KerasLayer(
#                 feature_extractor_url, 
#                 trainable=trainable
#             )
#         else:
#             self.feature_extractor = None

#     def build(self, input_shape):
#         # Make sure feature_extractor is initialized
#         if self.feature_extractor is None and self.feature_extractor_url:
#             self.feature_extractor = hub.KerasLayer(
#                 self.feature_extractor_url,
#                 trainable=self.trainable_setting
#             )
#         super().build(input_shape)

#     def call(self, inputs):
#         if self.feature_extractor is None:
#             raise ValueError("Feature extractor not initialized")
#         return self.feature_extractor(inputs)
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             'feature_extractor_url': self.feature_extractor_url,
#             'trainable': self.trainable_setting
#         })
#         return config
    
#     @classmethod
#     def from_config(cls, config):
#         # Extract the necessary configuration
#         url = config.pop('feature_extractor_url', None)
#         trainable = config.pop('trainable', False)
#         return cls(feature_extractor_url=url, trainable=trainable, **config)

# # Function to load the model
# @st.cache_resource
# def load_model():
#     custom_objects = {
#         "FeatureExtractorLayer": FeatureExtractorLayer
#     }
#     model = tf.keras.models.load_model("best_model_tf.h5", custom_objects=custom_objects)
#     return model

# # Load breed names
# @st.cache_data
# def load_breeds():
#     import pandas as pd
#     df = pd.read_csv("labels.csv")
#     breeds = sorted(df["breed"].unique())
#     return breeds

# # Function to fetch example breed images
# @st.cache_data
# def get_breed_example_image(breed_name):
#     """
#     Fetches an example image for the given breed from Dog API or falls back to local cache.
#     Returns the image as a PIL Image object.
#     """
#     # Create a cache directory if it doesn't exist
#     cache_dir = "breed_examples"
#     if not os.path.exists(cache_dir):
#         os.makedirs(cache_dir)
    
#     # Check if we have a cached image
#     cache_path = os.path.join(cache_dir, f"{breed_name.replace(' ', '_')}.jpg")
#     if os.path.exists(cache_path):
#         return Image.open(cache_path)
    
#     # Format breed name for API request (e.g., "German Shepherd" -> "german/shepherd")
#     formatted_breed = breed_name.lower().replace(' ', '/')
    
#     try:
#         # Try to fetch from Dog CEO API
#         response = requests.get(f"https://dog.ceo/api/breed/{formatted_breed}/images/random")
        
#         if response.status_code == 200:
#             data = response.json()
#             if data["status"] == "success":
#                 img_url = data["message"]
#                 img_response = requests.get(img_url)
#                 img = Image.open(BytesIO(img_response.content))
                
#                 # Cache the image for future use
#                 img.save(cache_path)
#                 return img
#     except Exception as e:
#         st.warning(f"Could not fetch example image: {e}")
    
#     # If API fails or image not found, return a placeholder image
#     placeholder = Image.new('RGB', (300, 300), color=(200, 200, 200))
#     return placeholder

# # Preprocess image
# def preprocess_image(image):
#     IMG_SIZE = 224
#     image = image.resize((IMG_SIZE, IMG_SIZE))
#     image = np.array(image) / 255.0
#     return np.expand_dims(image, axis=0)

# # Streamlit app
# st.title("Dog Breed Classifier")
# st.write("""
# Upload a photo of a dog, and this app will predict the breed and show you an example image of that breed.
# The model uses MobileNetV2 as a feature extractor and was trained on a dataset of dog images.
# """)

# # Add a sidebar with information
# with st.sidebar:
#     st.header("About")
#     st.write("""
#     This app uses a deep learning model trained to identify dog breeds.
    
#     The model can identify these top dog breeds:
#     - Labrador Retriever
#     - German Shepherd
#     - Golden Retriever
#     - Bulldog
#     - Poodle
#     - Beagle
#     - Rottweiler
#     - Yorkshire Terrier
#     - Boxer
#     And many more!
#     """)
    
#     st.header("How it works")
#     st.write("""
#     1. Upload an image of a dog
#     2. The model will analyze the image
#     3. You'll see the top 3 breed predictions
#     4. Example images of each predicted breed will be shown
#     """)


# model = load_model()
# breeds = load_breeds()

# uploaded_file = st.file_uploader("Choose a dog image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     # Preprocess and predict
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
    
#     # Get top 3 predictions
#     top_indices = np.argsort(prediction[0])[-3:][::-1]
    
#     st.write("## Predictions:")
    
#     # Create columns for predictions and example images
#     for i, idx in enumerate(top_indices):
#         breed_name = breeds[idx]
#         confidence = prediction[0][idx] * 100
        
#         col1, col2 = st.columns([1, 1])
#         with col1:
#             st.write(f"### {i+1}. {breed_name}")
#             st.write(f"Confidence: {confidence:.2f}%")
        
#         with col2:
#             # Display example image
#             with st.spinner(f"Loading example image for {breed_name}..."):
#                 try:
#                     example_img = get_breed_example_image(breed_name)
#                     st.image(example_img, caption=f"Example of {breed_name}", width=200)
#                 except Exception as e:
#                     st.error(f"Could not load example image: {e}")
        
#         st.divider()