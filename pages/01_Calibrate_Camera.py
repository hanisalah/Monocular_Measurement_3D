import streamlit as st
import cv2
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
from zipfile import ZipFile
from io import StringIO, BytesIO
from datetime import datetime
from src.cam_calib import calibrate_camera

def ix_change(mode=0):
    if mode==1:
        st.session_state.cam_calib_ix = min(len(st.session_state.cam_calib_imgs), st.session_state.cam_calib_ix + 1)
    elif mode==-1:
        st.session_state.cam_calib_ix = max(-1, st.session_state.cam_calib_ix - 1)

def calibrate():
    st.session_state.cam_calib_mx_str, st.session_state.cam_calib_mx = calibrate_camera(st.session_state.cam_calib_imgs, (st.session_state.ver,st.session_state.hor), st.session_state.sqSize)

def save_calib():
    zip_buf = BytesIO()
    with ZipFile(zip_buf, 'a') as file:
        for i in range(len(st.session_state.cam_calib_imgs)):
            _, img_buf = cv2.imencode('.png',st.session_state.cam_calib_imgs[i])
            file.writestr('cam_calib_imgs/{:06d}.png'.format(i), img_buf.tobytes())

        txt_buf = StringIO(st.session_state.cam_calib_mx_str)
        file.writestr('cam_calib_master_files/{}.txt'.format(st.session_state.cam_calib_file_name), txt_buf.getvalue())
    return zip_buf

if 'cam_calib_imgs' not in st.session_state.keys():
    st.session_state.cam_calib_imgs = []
if 'cam_calib_ix' not in st.session_state.keys():
    st.session_state.cam_calib_ix = -1
if 'cam_calib_mx' not in st.session_state.keys():
    st.session_state.cam_calib_mx = np.zeros((3,4),dtype=np.float32)
if 'tmp_calib_file_set' not in st.session_state.keys():
    st.session_state.tmp_calib_file_set =set()
if 'tmp_cam_dev_img_id' not in st.session_state.keys():
    st.session_state.tmp_cam_dev_img_id =''
if 'hor' not in st.session_state.keys():
    st.session_state.hor=0
if 'ver' not in st.session_state.keys():
    st.session_state.ver=0
if 'sqSize' not in st.session_state.keys():
    st.session_state.sqSize=0
if 'cam_calib_mx_str' not in st.session_state.keys():
    st.session_state.cam_calib_mx_str =''
if 'cam_calib_file_name' not in st.session_state.keys():
    st.session_state.cam_calib_file_name = ''

ip_method = st.radio('Input Method',('Camera','Files'))
if ip_method == 'Camera':
    img_stream = st.camera_input('Capture an image')
    if img_stream is not None:
        if img_stream.id != st.session_state.tmp_cam_dev_img_id:
            img_bytes = img_stream.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            st.session_state.cam_calib_imgs.append(cv_img)
            st.session_state.cam_calib_ix += 1
            st.session_state.tmp_cam_dev_img_id = img_stream.id
elif ip_method == 'Files':
    img_streams = st.file_uploader('Select Image', type=['png','jpeg','jpg'], key='calib_fu', accept_multiple_files=True)
    new_file_names = [i.name for i in img_streams]
    new_file_set = set(new_file_names)
    diff_file_set = new_file_set.difference(st.session_state.tmp_calib_file_set)
    for img_stream in img_streams:
        if img_stream.name in diff_file_set:
            img_bytes = img_stream.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            st.session_state.cam_calib_imgs.append(cv_img)
            st.session_state.cam_calib_ix += 1
            st.session_state.tmp_calib_file_set.add(img_stream.name)

st.session_state.ver = st.slider('Number of inner edges in a vertical line', min_value=0, max_value=11, value=9, step=1)
st.session_state.hor = st.slider('Number of inner edges in a horizontal line', min_value=0, max_value=11, value=7, step=1)
st.session_state.sqSize = st.number_input('Length of a square side in mm', min_value=0.0, value=24.0)
st.session_state.cam_calib_file_name = st.text_input('Calibration File Name', value='iphone_13')

if st.session_state.cam_calib_ix >= 0:
    st.image(st.session_state.cam_calib_imgs[st.session_state.cam_calib_ix], channels='BGR')

col41, col42, col43, col44, col45 = st.columns([2,1,2,2,2])
with col41:
    prev_dsbl = True if st.session_state.cam_calib_ix <= 0 else False
    is_prev = st.button('Prev', on_click=ix_change, kwargs={'mode':-1},disabled = prev_dsbl)
with col42:
    st.text(str(st.session_state.cam_calib_ix) if st.session_state.cam_calib_ix>=0 else '')
with col43:
    next_dsbl = True if st.session_state.cam_calib_ix >= len(st.session_state.cam_calib_imgs)-1 else False
    is_next = st.button('Next', on_click=ix_change, kwargs={'mode':1},disabled = next_dsbl)
with col44:
    calib_dsbl = True if len(st.session_state.cam_calib_imgs) <= 0 else False
    is_calib = st.button('Calibrate', on_click=calibrate, disabled = calib_dsbl)
with col45:
    save_dsbl = True if len(st.session_state.cam_calib_imgs)<=0 or st.session_state.cam_calib_mx_str=='' else False
    dt = '%Y-%m-%d-%H-%M-%S'
    is_save_all = st.download_button('Save All', save_calib(), file_name='calibration-'+datetime.now().strftime(dt)+'.zip', mime='application/x-zip', disabled = save_dsbl)

st.dataframe(st.session_state.cam_calib_mx)
