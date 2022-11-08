import sys
import subprocess
import torch
if torch.cuda.is_available():
    try:
        import iou3d_cuda
    except:
        import platform
        if platform.system() == 'Windows':
            subprocess.run([sys.executable,'-m', 'pip', 'install', './src/iou3d_win'])
        else:
            subprocess.run([sys.executable,'-m', 'pip', 'install', './src/iou3d_unix'])
        import importlib
        importlib.invalidate_caches()
        import sys
        import torch
        import iou3d_cuda
if torch.cuda.is_available():
    subprocess.run(['pip','install','opencv-python'])
else:
    subprocess.run(['pip','install','opencv-python-headless'])
if not torch.cuda.is_available():
    subprocess.run(['pip','install','streamlit-drawable-canvas','--upgrade'])
import streamlit as st
from pathlib import Path

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

st.title('3D Object Identification and Measurement')

is_clear = st.sidebar.button('Clear all data')
if is_clear:
    for key in st.session_state.keys():
        del st.session_state[key]

st.header('Info')
st.subheader('Main Page')
st.write("""
            From within this page you can:
            1. Read a brief description of functionality of each page.
            2. Clear all data stored during the session by clicking the button on the sidebar
            It is very important to note that a streamlit app can be activated on a local computer or through a web deployment.
            Training a model can only be executed locally since there is no access to a GPU through a deployed app.""")
st.subheader('Calibrate Camera Page')
st.write("""
            1. From this page you should upload calibration images, calibrate your camera, and store the results (calibration matrix).
            2. While there is an interface to capture images through camera, it is highly recommended to take images by your native app and use the upload files options instead.
               This is because the camera interface is not yet mature and forces very low resolution of captured images.
            3. To calibrate a single camera, you need around 12 to 20 images of the calibration chessboard pattern.
            4. Take the images such that the calibration pattern appear in different areas and different depths within each image.
            5. Specify the number of the inner edges both horizontally and vertically for your chessboard pattern using the given slidebars.
            6. Specify the length of a single square side in the calibration images in mm.
            7. Click the calibrate button and when calibration is done, click the save all button.""")
st.subheader('Label Dataset Page')
st.write("""
            1. From this page you should upload your training dataset images for labelling.
            2. While there is an interface to capture images through camera, it is highly recommended to take images by your native app and use the upload files options instead.
               This is because the camera interface is not yet mature and forces very low resolution of captured images.
            3. For each image you capture, click and drag to draw rectangles on objects of interest.
            4. When you finish drawing boxes in a single image, press the 'label' button. you will see a table for all labellings required.
            5. The coordinates of the bounding boxes will be automatically populated in the table you see, you manually have to enter the other data.
            6. When you finish labelling the dataset, specify the first file number and click the save all button.
            7. The program will create a zip file for all images and labels starting from the specified first file number, and the zip file will be saved to PC.""")
st.subheader('Real Time Measurement Page')
st.write("""
            1. This page is the main program interface. It includes two options shown in the side bar to either train a model or measure using trained model.
            2. Note that training such a model requires access to a decent GPU. This is why you shouldn't run this on a deployed app, but only on a local instance of the app.
            3. In deployed app, because there is no access to GPU, you will not see the selection in the sidebar and you will only be able to measure using a trained model.
            4. In training mode, you will be able to configure the settings using the shown interface to prepare the dataset and train the model.
            5. In measurement mode, you will be able to upload an image for detecting objects and calculating its dimensions.
            6. While there is an interface to capture images through camera, it is highly recommended to take images by your native app and use the upload files options instead.
               This is because the camera interface is not yet mature and forces very low resolution of captured images.
            7. Once prediction images are uploaded, press the compute button and the model will predict and draw 3D bounding boxes around objects and provide a table showing the dimensions.""")
st.subheader('About Page')
st.write("""1. This page provides more technical details about the program and the above functionalities""")
