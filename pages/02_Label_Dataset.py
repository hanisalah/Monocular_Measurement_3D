import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from zipfile import ZipFile
from io import StringIO, BytesIO
from datetime import datetime
from PIL import Image
from st_aggrid import AgGrid
from copy import deepcopy

def label(result=None, ix=-1):
    if result is not None:
        objects = pd.json_normalize(result["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        #st.dataframe(objects)
        if len(objects)!=0 and ix>=0:
            st.session_state.df_anns[ix]['labels'] = pd.DataFrame({'class_name':'Empty', 'truncation':0, 'occlusion': 0, 'alpha':0,
                                                                    'bbox_tl_x':(objects['left']*scale_w).astype('int').tolist(),
                                                                    'bbox_tl_y':(objects['top']*scale_h).astype('int').tolist(),
                                                                    'bbox_br_x':((objects['left']+objects['width'])*scale_w).astype('int').tolist(),
                                                                    'bbox_br_y':((objects['top']+objects['height'])*scale_h).astype('int').tolist(),
                                                                    'height':0, 'width':0, 'length':0, 'loc_x':0, 'loc_y':0, 'loc_z':0, 'rot_y':0})

def ix_change(mode=0):
    if mode==1:
        st.session_state.df_ix = min(len(st.session_state.df_anns), st.session_state.df_ix + 1)
    elif mode==-1:
        st.session_state.df_ix = max(-1, st.session_state.df_ix - 1)

def save_dataset(offset=0):
    zip_buf = BytesIO()
    with ZipFile(zip_buf, 'a') as file:
        for i in range(len(st.session_state.df_anns)):
            _, img_buf = cv2.imencode('.png',st.session_state.df_anns[i]['img'])
            file.writestr('imgs/{:06d}.png'.format(i+offset), img_buf.tobytes())
            if len(st.session_state.df_anns[i]['labels']) > 0:
                txt_buf = StringIO()
                st.session_state.df_anns[i]['labels'].to_csv(txt_buf, sep=' ', header=False, index=False)
                file.writestr('labels/{:06d}.txt'.format(i+offset), txt_buf.getvalue())
    return zip_buf

df_columns = ['class_name','truncation','occlusion','alpha','bbox_tl_x','bbox_tl_y','bbox_br_x','bbox_br_y','height','width','length','loc_x','loc_y','loc_z','rot_y']
entry = {'img':'', 'labels': pd.DataFrame(columns=df_columns)}

if 'df_anns' not in st.session_state.keys():
    st.session_state.df_anns = []
if 'df_ix' not in st.session_state.keys():
    st.session_state.df_ix = -1
if 'tmp_ds_file_set' not in st.session_state.keys():
    st.session_state.tmp_ds_file_set =set()
if 'tmp_cam_dev_img_id' not in st.session_state.keys():
    st.session_state.tmp_cam_dev_img_id =''

ip_method = st.radio('Input Method',('Camera','Files'))
if ip_method == 'Camera':
    img_stream = st.camera_input('Capture an image')
    if img_stream is not None:
        if img_stream.id != st.session_state.tmp_cam_dev_img_id:
            img_bytes = img_stream.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            new_entry = entry.copy()
            new_entry['img']=cv_img
            st.session_state.df_anns.append(new_entry)
            st.session_state.df_ix += 1
            st.session_state.tmp_cam_dev_img_id = img_stream.id
elif ip_method == 'Files':
    img_streams = st.file_uploader('Select Image', type=['png','jpeg','jpg'], key='dset_fu', accept_multiple_files=True)
    new_file_names = [i.name for i in img_streams]
    new_file_set = set(new_file_names)
    diff_file_set = new_file_set.difference(st.session_state.tmp_ds_file_set)
    for img_stream in img_streams:
        if img_stream.name in diff_file_set:
            img_bytes = img_stream.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            new_entry = entry.copy()
            new_entry['img']=cv_img
            st.session_state.df_anns.append(new_entry)
            st.session_state.df_ix += 1
            st.session_state.tmp_ds_file_set.add(img_stream.name)

#img_org_w = st.session_state.df_anns[st.session_state.df_ix]['img'].shape[1] if st.session_state.df_ix >= 0 else 0
#img_org_h = st.session_state.df_anns[st.session_state.df_ix]['img'].shape[0] if st.session_state.df_ix >= 0 else 0
#cvs_w = 700
#cvs_h = int(img_org_h * (cvs_w/img_org_w)) if st.session_state.df_ix >= 0 else 400
#scale_w = img_org_w / cvs_w if st.session_state.df_ix >= 0 else 1
#scale_h = img_org_h / cvs_h if st.session_state.df_ix >= 0 else 1
#scale_img = cv2.resize(st.session_state.df_anns[st.session_state.df_ix]['img'][:,:,::-1], (cvs_w,cvs_h)) if st.session_state.df_ix >= 0 else None
#if st.session_state.df_ix >= 0:
    #x = Image.fromarray(scale_img)
    #img_obj = BytesIO()
    #x.save(img_obj,format='png')
    #img_obj.seek(0)
#else:
    #img_obj=None

#canvas_result = st_canvas(fill_color='rgba(0,165,255,0.3)', stroke_width=3, stroke_color='#000000', background_color='#eee',
                            #background_image=Image.open(img_obj) if img_obj else None, update_streamlit=True, height=cvs_h, width=cvs_w,
                            #drawing_mode='rect', point_display_radius=0, key='canvas'+(str(st.session_state.df_ix) if st.session_state.df_ix>=0 else ''))


if st.session_state.df_ix >=0:
    img = deepcopy(st.session_state.df_anns[st.session_state.df_ix]['img'])
    index = deepcopy(st.session_state.df_ix)
    img_org_w = img.shape[1]
    img_org_h = img.shape[0]
    cvs_w=700
    cvs_h = int(img_org_h * (cvs_w/img_org_w))
    scale_w = img_org_w / cvs_w
    scale_h = img_org_h / cvs_h
    scale_img = cv2.resize(img[:,:,::-1], (cvs_w,cvs_h))
    x = Image.fromarray(scale_img)
    img_obj = BytesIO()
    x.save(img_obj, format='png')
    img_obj.seek(0)
    canvas_key = 'canvas'+str(index)
else:
    cvs_w = 700
    cvs_h = 400
    img_obj=None
    canvas_key = 'canvas'

canvas_result = st_canvas(fill_color='rgba(0,165,255,0.3)', stroke_width=3, stroke_color='#000000', background_color='#eee',
                            background_image=Image.open(img_obj) if img_obj else None, update_streamlit=True, height=cvs_h, width=cvs_w,
                            drawing_mode='rect', point_display_radius=0, key=canvas_key)

col41, col42, col43, col44, col45, col46 = st.columns([2,1,2,2,2,2])
with col41:
    prev_dsbl = True if st.session_state.df_ix <= 0 else False
    is_prev = st.button('Prev', on_click=ix_change, kwargs={'mode':-1},disabled = prev_dsbl)
with col42:
    st.text(str(st.session_state.df_ix) if st.session_state.df_ix>=0 else '')
with col43:
    next_dsbl = True if st.session_state.df_ix >= len(st.session_state.df_anns)-1 else False
    is_next = st.button('Next', on_click=ix_change, kwargs={'mode':1},disabled = next_dsbl)
with col44:
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        push_dsbl = True if len(objects)==0 else False
    else:
        push_dsbl = True
    is_lbl = st.button('Label', on_click=label, kwargs={'result': canvas_result.json_data, 'ix':st.session_state.df_ix}, disabled = push_dsbl)
with col45:
    save_dsbl = True if len(st.session_state.df_anns)<=0 else False
    offset = int(st.number_input('First File no.',min_value=0, max_value=999999, value=0, step=1, disabled=save_dsbl))
with col46:
    dt = '%Y-%m-%d-%H-%M-%S'
    is_save_all = st.download_button('Save All', save_dataset(int(offset)), file_name='dataset-'+datetime.now().strftime(dt)+'.zip', mime='application/x-zip', disabled = save_dsbl)

if st.session_state.df_ix >=0 :
    if len(st.session_state.df_anns[st.session_state.df_ix]['labels']) == 0:
        df = pd.DataFrame(columns=df_columns)
        df.loc[len(df.index)] = ['Empty',0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        st.dataframe(df)
    else:
        grid_return = AgGrid(st.session_state.df_anns[st.session_state.df_ix]['labels'], key='default', reload_data=False, editable =True)
        st.session_state.df_anns[st.session_state.df_ix]['labels'] = pd.DataFrame(grid_return['data'])
