import os
from copy import deepcopy
from io import StringIO, BytesIO
from datetime import datetime
import glob
from zipfile import ZipFile
import numpy as np
import pandas as pd
import cv2
import time

import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

from src.prep_data import get_cats, det_cats, populate_calib, prepare_data, lockers
from src.RTM3D_main import runny
import src.redirect as rd
import torch

def ix_change(mode=0):
    if mode==1:
        st.session_state.predict_img_ix = min(len(st.session_state.predict_img), st.session_state.predict_img_ix + 1)
    elif mode==-1:
        st.session_state.predict_img_ix = max(-1, st.session_state.predict_img_ix - 1)

def read_files(path):
    files = glob.glob(path)
    if len(files) != 0:
        files = [i.split('/')[-1] for i in files]
        files = [i.split('\\')[-1] for i in files]
        files = [i.split('.')[0] for i in files]
    return files

def compute(paths, settings, img_ix, calib_master_path):
    so = st.empty()
    with rd.stdout(to=so):
        image = st.session_state.predict_img[img_ix]['img']
        result = runny('predict', paths, settings, image, calib_master_path)
        st.session_state.predict_img[img_ix]['img_op'] = deepcopy(result['img'])
        st.session_state.predict_img[img_ix]['labels'] = pd.DataFrame(deepcopy(result['dict']))
        time.sleep(5)

def save_results():
    zip_buf = BytesIO()
    with ZipFile(zip_buf, 'a') as file:
        for i in range(len(st.session_state.predict_img)):
            _, img_buf = cv2.imencode('.png',st.session_state.predict_img[i]['img'])
            file.writestr('imgs/{}.png'.format(st.session_state.predict_img[i]['img_name']), img_buf.tobytes())
            if st.session_state.predict_img[i]['img_op']!='':
                _, img_op_buf = cv2.imencode('.png',st.session_state.predict_img[i]['img_op'])
                file.writestr('imgs/{}_Boxed.png'.format(st.session_state.predict_img[i]['img_name']), img_op_buf.tobytes())
            if len(st.session_state.predict_img[i]['labels']) > 0:
                txt_buf = StringIO()
                st.session_state.predict_img[i]['labels'].to_csv(txt_buf, sep=' ', header=True, index=False)
                file.writestr('labels/{}_Measurement.txt'.format(st.session_state.predict_img[i]['img_name']), txt_buf.getvalue())
    return zip_buf

app_path = '/'.join(os.path.abspath(os.path.dirname(__file__)).split('\\')[:-1])
st.write('path:'+app_path)
if app_path=='':
    app_path = '/'.join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1])#'/app/monocular_measurement_3d'
st.write('path:'+app_path)
paths = {'root_path' : app_path, 'data_dir' : app_path+'/train_data/', 'image_dir' : app_path + '/train_data/dset_imgs/', 'calib_dir' : app_path + '/train_data/dset_calib/',
        'label_dir' : app_path + '/train_data/dset_labels/', 'working_dir' : app_path + '/train_data/anns/', 'demo' : app_path + '/train_data/predictions/',
        'exp_dir' : app_path +'/train_data/model/exp/', 'save_dir' : app_path +'/train_data/model/exp/default/', 'debug_dir' : app_path + '/train_data/model/exp/debug/',
        'results_dir' : app_path + '/train_data/model/exp/results/', 'model_dir' : app_path +'/train_data/model/exp/default/','master_calib': app_path+'/train_data/cam_calib_master_files/'}
settings = {'det_cats':det_cats(paths['working_dir']), 'is_resume':False, 'load_model':'', 'batch_size':int(2), 'locker_sizes':lockers(paths['working_dir'])}
paths_help = {
                'root_path':'This is the root directory of the application',
                'data_dir':'This is the root directory of the training dataset',
                'image_dir':'This is the directory containing images of the training dataset',
                'calib_dir':'This is the directory containing calibration file for each image of the training dataset',
                'label_dir':'This is the directory containing labels of the training dataset',
                'working_dir':'This is the directory containing labels in JSON format as well as dataset statistics, category names..etc',
                'demo':'This is the directory containing prediction images for testing',
                'exp_dir':'For internal use',
                'save_dir':'For internal use',
                'debug_dir':'For internal use',
                'results_dir':'For internal use',
                'model_dir':'This is the directory containing the model .pth files',
                'master_calib':'This is the directory containing the master calibration files. 1 file for each camera'}

#for k,v in paths.items():
    #if not os.path.exists(v):
        #os.makedirs(v)

df_columns = ['Class', 'alpha', 'bbox[0]', 'bbox[1]', 'bbox[2]', 'bbox[3]', 'h', 'w', 'l', 'Px', 'Py', 'Pz', 'ori', 'score', 'locker_1', 'locker_2', 'locker_3']
entry = {'img':'', 'labels': pd.DataFrame(columns=df_columns), 'img_op':'', 'img_name':''}
if 'predict_img' not in st.session_state.keys():
    st.session_state.predict_img = []
if 'predict_img_ix' not in st.session_state.keys():
    st.session_state.predict_img_ix = -1
if 'tmp_predict_file_set' not in st.session_state.keys():
    st.session_state.tmp_predict_file_set =set()
if 'tmp_cam_dev_img_id' not in st.session_state.keys():
    st.session_state.tmp_cam_dev_img_id =''

train_enable = True if torch.cuda.is_available() else False
if train_enable:
    flag = st.sidebar.radio('',['Train','Measure'], index=0, key='train_measure_radio')
else:
    flag = 'Measure'

#All training functions go here
if train_enable == True and flag =='Train':
    st.info('These are the paths the program will use.')
    paths_df = pd.DataFrame([paths_help, paths]).T
    paths_df.reset_index(inplace=True)
    paths_df.columns=['Variable','Description','Path']
    st.dataframe(paths_df, width=2000, height=500)
    st.info('Copy Dataset images to {}'.format(paths['image_dir']))
    st.info('Copy Dataset Labels to {}'.format(paths['label_dir']))
    st.info('Copy Master Calibration Files to {}'.format(paths['master_calib']))

    calibs = read_files(app_path+'/train_data/cam_calib_master_files/*.txt')
    seld = st.selectbox('Select Camera Calibration File',calibs, key='train_master_calib_selectbox')

    cats = get_cats(paths['working_dir'], paths['label_dir'])
    det_cats_ms_label = 'Choose the Categories you want to include in training.'
    det_cats_list = st.multiselect(det_cats_ms_label, cats, default= settings['det_cats'], key='train_det_cats_multiselect')

    st.write('Locker Sizes: Define Locker Sizes in meters')
    locker_df = pd.DataFrame(settings['locker_sizes'], columns=['dim1','dim2','dim3'])
    gb = GridOptionsBuilder.from_dataframe(locker_df)
    gb.configure_default_column(editable=True)
    string_to_add_row = '''
                        function(e) {
                                    let api = e.api;
                                    let rowIndex = e.rowIndex + 1;
                                    api.applyTransaction({addIndex: rowIndex, add: [{}]});
                                    };
                        '''
    cell_button_add = JsCode('''
                                class BtnAddCellRenderer {
                                    init(params) {
                                        this.params = params;
                                        this.eGui = document.createElement('div');
                                        this.eGui.innerHTML = `
                                         <span>
                                            <style>
                                            .btn_add {
                                              background-color: limegreen;
                                              border: none;
                                              color: white;
                                              text-align: center;
                                              text-decoration: none;
                                              display: inline-block;
                                              font-size: 10px;
                                              font-weight: bold;
                                              height: 2.5em;
                                              width: 8em;
                                              cursor: pointer;
                                            }

                                            .btn_add :hover {
                                              background-color: #05d588;
                                            }
                                            </style>
                                            <button id='click-button'
                                                class="btn_add"
                                                >&CirclePlus; Add</button>
                                         </span>
                                      `;
                                    }
                                    getGui() {
                                        return this.eGui;
                                    }
                                };
                                ''')
    gb.configure_column('Add', headerTooltip='Click on Button to add new row', editable=False, filter=False, onCellClicked=JsCode(string_to_add_row),
                        cellRenderer=cell_button_add, autoHeight=True, wrapText=True, lockPosition='left')
    string_to_delete = '''
                        function(e) {
                                    let api = e.api;
                                    let sel = api.getSelectedRows();
                                    api.applyTransaction({remove: sel});
                                    };
                        '''
    cell_button_delete = JsCode('''
                                class BtnCellRenderer {
                                    init(params) {
                                        console.log(params.api.getSelectedRows());
                                        this.params = params;
                                        this.eGui = document.createElement('div');
                                        this.eGui.innerHTML = `
                                         <span>
                                            <style>
                                            .btn {
                                              background-color: #F94721;
                                              border: none;
                                              color: white;
                                              font-size: 10px;
                                              font-weight: bold;
                                              height: 2.5em;
                                              width: 8em;
                                              cursor: pointer;
                                            }

                                            .btn:hover {
                                              background-color: #FB6747;
                                            }
                                            </style>
                                            <button id='click-button2'
                                                class="btn"
                                                >&#128465; Delete</button>
                                         </span>
                                      `;
                                    }

                                    getGui() {
                                        return this.eGui;
                                    }
                                };
                                ''')
    gb.configure_column('Delete', headerTooltip='Select using Checkbox then Click on Button to remove row', editable=False, filter=False, onCellClicked=JsCode(string_to_delete),
                        cellRenderer=cell_button_delete, autoHeight=True, suppressMovable='true', checkboxSelection=True)
    gridOptions = gb.build()
    locker_return = AgGrid(locker_df, gridOptions=gridOptions, key='locker_aggrid', reload_data=False, editable =True, fit_columns_on_grid_load=True, allow_unsafe_jscode=True)
    locker_df = locker_return['data']
    locker_list = locker_df.values.tolist()

    calib_string = 'Copy selected master calibration file such that for each dataset image, there is a calibration text file with the same name as the dataset image.'
    det_cats_string = 'Save the shown categories to be those categories that will be used for training and prediction.'
    locker_string = 'Save the shown locker sizes to be used in measurement to select the smallest locker that the detected object will fit in.'
    json_string = 'Convert the dataset images, labels and calibrations to json format to be used in training.'
    calib_check = st.checkbox(calib_string,key='train_master_calib_checkbox')
    det_cats_check = st.checkbox(det_cats_string, key='train_det_cats_checkbox')
    locker_sizes_check = st.checkbox(locker_string, key='train_locker_sizes_checkbox')
    prepare_json_data_check = st.checkbox(json_string, key='train_generate_data_checkbox')


    c1, c2 = st.columns([2,4])
    models = read_files(paths['model_dir']+'*.pth')
    is_model_dsbl = True if len(models)==0 else False
    with c1:
        resume_radio = st.radio('',['New Model','Resume Existing Model'],index=0,disabled=is_model_dsbl, key='train_model_select_radio')
        settings['is_resume']= True if resume_radio == 'Resume Existing Model' else False
    with c2:
        is_load_dsbl = True if is_model_dsbl or resume_radio=='New Model' else False
        load_model = st.selectbox('Load Model to resume its training',models, disabled=is_load_dsbl, key='train_model_load_selectbox')
        if load_model is not None and is_load_dsbl==False:
            opts_load_model = load_model + '.pth'
        settings['load_model']= (paths['model_dir'] + opts_load_model) if settings['is_resume'] else ''

    settings['batch_size']= st.slider('Define Batch Size', min_value=2, max_value=32, value=2, step=1, key='train_batch_size_slider')

    _,c3, c4,_ = st.columns(4)
    with c3:
        is_prep_data = st.button('Prepare Data', key='train_prepare_data_button')
    with c4:
        is_train = st.button('Train', key='train_button')

    so1 = st.empty()
    with rd.stdout(to=so1): #redirect print statments in any function (called within with statement) to streamlit container.
        if is_prep_data:
            if calib_check:
                print('Populating Calibration Files...')
                populate_calib(app_path,seld)
                print('Populating Calibration Files completed!')
            if det_cats_check:
                print('Generating detected categories...')
                settings['det_cats'] = det_cats(paths['working_dir'], det_cats_list)
                print('Generating detected categories completed!')
            if locker_sizes_check:
                print('Generating locker sizes...')
                settings['locker_sizes'] = lockers(paths['working_dir'],locker_list)
                print('Generating locker sizes completed!')
            if prepare_json_data_check:
                print('Preparing dataset...')
                prepare_data(app_path, settings['det_cats'])
                print('Preparing dataset completed!')
            print('Preparing Data Completed. You can train!')
        if is_train:
            p = deepcopy(paths)
            s = deepcopy(settings)
            runny('train',p,s)

#All prediction functions go here
if flag == 'Measure':
    models = read_files(paths['model_dir']+'*.*')#'*.pth')
    model_seld = st.selectbox('Select Model File',models)
    if model_seld is not None:
        opts_load_model = model_seld + '.pth'
    else:
        opts_load_model=''
    settings['load_model']= paths['model_dir'] + opts_load_model

    calibs = read_files(app_path+'/train_data/cam_calib_master_files/*.txt')
    cal_seld = st.selectbox('Select Camera Calibration File',calibs)

    ip_method = st.radio('Input Method',('Camera','Files'))
    if ip_method == 'Camera':
        img_stream = st.camera_input('Capture an image')
        if img_stream is not None:
            if img_stream.id != st.session_state.tmp_cam_dev_img_id:
                img_bytes = img_stream.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                new_entry = entry.copy()
                new_entry['img']=cv_img
                new_entry['img_name'] = img_stream.name
                st.session_state.predict_img.append(new_entry)
                st.session_state.predict_img_ix += 1
                st.session_state.tmp_cam_dev_img_id = img_stream.id
    elif ip_method == 'Files':
        img_streams = st.file_uploader('Select Image', type=['png','jpeg','jpg'], key='predict_fu', accept_multiple_files=True)
        new_file_names = [i.name for i in img_streams]
        new_file_set = set(new_file_names)
        diff_file_set = new_file_set.difference(st.session_state.tmp_predict_file_set)
        for img_stream in img_streams:
            if img_stream.name in diff_file_set:
                img_bytes = img_stream.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                new_entry = entry.copy()
                new_entry['img']=cv_img
                new_entry['img_name']=img_stream.name
                st.session_state.predict_img.append(new_entry)
                st.session_state.predict_img_ix += 1
                st.session_state.tmp_predict_file_set.add(img_stream.name)

    if st.session_state.predict_img_ix >= 0:
        st.image(st.session_state.predict_img[st.session_state.predict_img_ix]['img'], channels='BGR')

    col41, col42, col43, col44, col45 = st.columns([2,1,2,2,2])
    with col41:
        prev_dsbl = True if st.session_state.predict_img_ix <= 0 else False
        is_prev = st.button('Prev', on_click=ix_change, kwargs={'mode':-1},disabled = prev_dsbl)
    with col42:
        st.text(str(st.session_state.predict_img_ix) if st.session_state.predict_img_ix>=0 else '')
    with col43:
        next_dsbl = True if st.session_state.predict_img_ix >= len(st.session_state.predict_img)-1 else False
        is_next = st.button('Next', on_click=ix_change, kwargs={'mode':1},disabled = next_dsbl)
    with col44:
        compute_dsbl = True if len(st.session_state.predict_img) <=0 else False
        is_compute = st.button('Compute', on_click=compute, args=(paths, settings, st.session_state.predict_img_ix, paths['master_calib']+cal_seld+'.txt'), disabled = compute_dsbl)
    with col45:
        save_dsbl = True if len(st.session_state.predict_img)<=0 else False
        dt = '%Y-%m-%d-%H-%M-%S'
        is_save_all = st.download_button('Save All', save_results(), file_name='measurements-'+datetime.now().strftime(dt)+'.zip', mime='application/x-zip', disabled = save_dsbl)

    if st.session_state.predict_img_ix >=0:
        if st.session_state.predict_img[st.session_state.predict_img_ix]['img_op']=='':
            st.image(np.full((480,640,3),128, dtype=np.uint8))
        else:
            st.image(st.session_state.predict_img[st.session_state.predict_img_ix]['img_op'], channels='BGR')

        if len(st.session_state.predict_img[st.session_state.predict_img_ix]['labels']) == 0:
            df = pd.DataFrame(columns=df_columns)
            df.loc[len(df.index)] = ['Empty',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            st.dataframe(df)
        else:
            st.dataframe(st.session_state.predict_img[st.session_state.predict_img_ix]['labels'])
