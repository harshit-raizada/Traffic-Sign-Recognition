import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('final_model_updated.h5')

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }  

def classify_image(image, model):
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis = 0)
    pred = model.predict(image)
    pred_class = np.argmax(pred, axis = 1)[0]
    return classes[pred_class]

st.markdown("""
<style>
body {
    color: #fff;
    background-color: #4F8BF9;
}
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.header-font {
    font-size:24px !important;
    color: #FFAE42;
}
.streamlit-expanderHeader {
    font-size: 20px;
    color: #FF6347;
}
.streamlit-expanderContent {
    font-size: 18px;
}
.css-2trqyj {
    background-color: #FF6347;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Traffic Sign Recognition App 🚦</p>', unsafe_allow_html = True)
st.markdown('<p class="header-font">Upload a traffic sign image and check its meaning. 🚥</p>', unsafe_allow_html = True)

uploaded_file = st.file_uploader("Choose an Image...", type = ["jpg", "png", "jpeg"], key = "file_uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = 'Uploaded Image', use_column_width = True)
    st.markdown('<p class="header-font">Classifying...</p>', unsafe_allow_html = True)
    with st.spinner('Processing...'):
        label = classify_image(image, model)
    st.success(f'Prediction: {label}')