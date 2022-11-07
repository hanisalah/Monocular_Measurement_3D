import numpy as np
import math
import os
import cv2
import torch
import json
import random
import pycocotools.coco as coco
from src.utils import compute_box_3d, project_to_image, project_to_image3,alpha2rot_y
from src.utils import draw_box_3d, unproject_2d_to_3d
import glob
import shutil

def _bbox_to_coco_bbox(bbox):
    return [(bbox[0]), (bbox[1]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib

def get_cats(annPathFinal, labelPathRaw):
    cats = set()
    if os.path.isfile(annPathFinal+'categories.txt'):
        with open(annPathFinal+'categories.txt','r') as f:
            lines = f.readlines()
            cats_list = [i[:-1] for i in lines]
    else:
        labelFileList = glob.glob(labelPathRaw+'*.txt')
        for labelFile in labelFileList:
            with open(labelFile,'r') as f:
                lines = f.readlines()
                for line in lines:
                    cats.add(line.split(' ')[0])
        cats_list = list(cats)
    if os.path.isfile(annPathFinal+'categories.txt') == False:
        with open(annPathFinal+'categories.txt','w') as f:
            s = '\n'.join(cats_list)
            s += '\n'
            f.write(s)
    return cats_list

def det_cats(annPathFinal, cats=[]):
    if len(cats)>0:
        with open(annPathFinal+'categories.txt','r') as f:
            lines = f.readlines()
            all_cats = [i[:-1] for i in lines]
        tmp = cats.copy()
        tmp.extend([x for x in all_cats if x not in tmp])
        with open(annPathFinal+'categories.txt','w')as f:
            s = '\n'.join(tmp)
            s += '\n'
            f.write(s)
        with open(annPathFinal+'det_cats.txt','w') as f:
            s = '\n'.join(cats)
            s += '\n'
            f.write(s)
        return cats

    elif os.path.isfile(annPathFinal+'det_cats.txt') and len(cats)==0:
        with open(annPathFinal+'det_cats.txt', 'r') as f:
            lines = f.readlines()
            det_cats_list = [i[:-1] for i in lines]
        return det_cats_list
    else:
        return []

def lockers(annPathFinal, locker_sizes=[]):
    if os.path.isfile(annPathFinal+'locker_sizes.txt') and len(locker_sizes)==0:
        with open(annPathFinal+'locker_sizes.txt', 'r') as f:
            lines = f.readlines()
            locker_sizes = [[float(x) for x in l.split(' ') ] for l in lines]
        return locker_sizes
    elif len(locker_sizes)>0:
        with open(annPathFinal+'locker_sizes.txt', 'w') as f:
            s=''
            for locker in locker_sizes:
                locker_string = ' '.join([str(x) for x in locker])
                s+= locker_string
                s+='\n'
            #s +='\n'
            f.write(s)
        return locker_sizes
    else:
        return [[0,0,0]]

def populate_calib(app_path, master_file):
    labelPath = app_path+'/train_data/dset_labels/'
    calibPath = app_path+'/train_data/dset_calib/'
    masterPath = app_path+'/train_data/cam_calib_master_files/'+master_file+'.txt'
    labelFiles = glob.glob(labelPath+'*.txt')
    labelFiles = [i.split('/')[-1] for i in labelFiles]
    labelFiles = [i.split('\\')[-1] for i in labelFiles]
    for labelFile in labelFiles:
        shutil.copy(masterPath, calibPath+labelFile)

def prepare_data(app_path, st_det_cats):
    #app_path ='/'.join(os.path.abspath('__file__').split('\\')[:-2])
    data_path = app_path + '/train_data/' #'/data/kitti/'
    imgPathRaw = data_path + '/dset_imgs/' #'data_object_image_2/training/image_2/'
    calibPathRaw = data_path + '/dset_calib/' #'data_object_calib/training/calib/'
    labelPathRaw = data_path + '/dset_labels/' #'data_object_label_2/training/label_2/'
    annPathFinal = data_path + '/anns/' #'/working/anns/'
    cats = get_cats(annPathFinal, labelPathRaw)
    det_cats = st_det_cats.copy()
    #det_cats = get_det_cats()
    #det_cats=['Car', 'Pedestrian', 'Cyclist'] #These are the categories that will be detected. They should be the same as opt.det_cats in case of custom dataset
    #cats = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare'] #These are all categories in dataset. They can be same or bigger than det_cats.

    if not os.path.exists(annPathFinal):
        os.makedirs(annPathFinal)
    isTrainAvailable = os.path.isfile(annPathFinal+'ann_train.json')
    isValAvailable = os.path.isfile(annPathFinal+'ann_val.json')
    isTestAvailable = os.path.isfile(annPathFinal+'ann_test.json')
    isStatsAvailable = os.path.isfile(annPathFinal+'stats.txt')

    if not isTrainAvailable or not isValAvailable:
        cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
        cat_info = []
        for i, cat in enumerate(cats):
            cat_info.append({'name': cat, 'id': i + 1})

        ret_train = {'images': [], 'annotations': [], "categories": cat_info}
        ret_val = {'images': [], 'annotations': [], "categories": cat_info}
        ret_test = {'images': [], 'annotations': [], "categories": cat_info}

        image_set =  os.listdir(imgPathRaw)
        image_set = [i.split('.')[0] for i in image_set]

        val_count = int(0.1 * len(image_set))
        test_count = int(0.1 * len(image_set))
        training_count = len(image_set) - val_count - test_count
        test_set = random.sample(image_set, k = test_count)
        files_wo_test_set = [f for f in image_set if f not in test_set]
        validation_set = random.sample(files_wo_test_set, k = val_count)
        training_set = [f for f in files_wo_test_set if f not in validation_set]

        image_to_id = {}
        print('Preparing JSON Files...')
        for line in image_set:
            #print('file {} of {}'.format(int(line), int(len(image_set))))#, end='\x1b[1K\r')
            image_id = int(line)
            calib_path = calibPathRaw  + '{}.txt'.format(line)
            calib = read_clib(calib_path)
            image_info = {'file_name': '{}.png'.format(line), 'id': int(image_id), 'calib': calib.tolist()}
            if line in training_set:
                ret_train['images'].append(image_info)
            elif line in validation_set:
                ret_val['images'].append(image_info)
            elif line in test_set:
                ret_test['images'].append(image_info)
            ann_path = labelPathRaw + '{}.txt'.format(line)
            anns = open(ann_path, 'r')
            for ann_ind, txt in enumerate(anns):
                tmp = txt[:-1].split(' ')
                cat_id = cat_ids[tmp[0]]
                truncated = int(float(tmp[1]))
                occluded = int(tmp[2])
                alpha = float(tmp[3])
                dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
                location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
                rotation_y = float(tmp[14])
                num_keypoints = 0
                box_2d_as_point=[0]*27
                bbox=[0.,0.,0.,0.]
                calib_list = np.reshape(calib, (12)).tolist()
                if tmp[0] in det_cats:
                    image = cv2.imread(imgPathRaw+ str(image_info['file_name'])) #(os.path.join(imgPathRaw, image_info['file_name']))
                    bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
                    box_3d = compute_box_3d(dim, location, rotation_y)
                    box_2d_as_point,vis_num,pts_center = project_to_image(box_3d, calib,image.shape)
                    box_2d_as_point=np.reshape(box_2d_as_point,(1,27))
                    box_2d_as_point=box_2d_as_point.tolist()[0]
                    num_keypoints=vis_num
                    alpha = rotation_y - math.atan2(pts_center[0, 0] - calib[0, 2], calib[0, 0])
                    if line in training_set:
                        id = int(len(ret_train['annotations']) +1)
                    elif line in validation_set:
                        id = int(len(ret_val['annotations']) +1)
                    elif line in test_set:
                        id = int(len(ret_test['annotations']) +1)
                    ann = {'segmentation': [[0,0,0,0,0,0]],
                           'num_keypoints':num_keypoints,
                           'area':1,
                           'iscrowd': 0,
                           'keypoints': box_2d_as_point,
                           'image_id': image_id,
                           'bbox': _bbox_to_coco_bbox(bbox),
                           'category_id': cat_id,
                           'id': id,
                           'dim': dim,
                           'rotation_y': rotation_y,
                           'alpha': alpha,
                           'location':location,
                           'calib':calib_list,
                            }
                    if line in training_set:
                        ret_train['annotations'].append(ann)
                    elif line in validation_set:
                        ret_val['annotations'].append(ann)
                    elif line in test_set:
                        ret_test['annotations'].append(ann)
        print("# images: ", len(ret_train['images']) + len(ret_val['images']) + len(ret_test['images']) )
        print("# annotations: ", len(ret_train['annotations']) + len(ret_val['annotations']) + len(ret_test['annotations']) )
        train_path = annPathFinal + 'ann_train.json'
        val_path = annPathFinal + 'ann_val.json'
        test_path = annPathFinal + 'ann_test.json'
        json.dump(ret_train, open(train_path, 'w'))
        json.dump(ret_val, open(val_path, 'w'))
        json.dump(ret_test, open(test_path, 'w'))
        print('Preparing JSON files completed!')

    if not isStatsAvailable:
        mean = []
        std = []
        h = []
        w = []
        c = coco.COCO(annPathFinal+'ann_train.json')
        image_ids = c.getImgIds()
        print('Preparing dataset statistics...')
        for image_id in image_ids:
            #print('file {} of {}'.format(int(image_id), int(len(image_ids))))#, end='\x1b[1K\r')
            file = c.loadImgs(ids=[image_id])[0]['file_name']
            img = cv2.imread(imgPathRaw+str(file))
            img = img/255
            mean.append(img.mean(axis=(0,1)))
            std.append(img.std(axis=(0,1)))
            h.append(img.shape[0])
            w.append(img.shape[1])
        mean = np.array(mean)
        std = np.array(std)
        mean = np.mean(mean, axis=0).tolist()
        std = np.mean(std, axis=0).tolist()
        h = int(max(h))
        w = int(max(w))
        h = int(h + (32 - (h%32))) #make h divisible by 32 to pass through model
        w = int(w + (32 - (w%32))) #make w divisible by 32 to pass through model
        stats = 'default_resolution {} {}\n'.format(str(h), str(w))
        stats += 'num_classes {}\n'.format(str(len(det_cats)))
        stats += 'mean {}\n'.format(' '.join(str(item) for item in mean))
        stats += 'std {}\n'.format(' '.join(str(item) for item in std))
        with open(annPathFinal+'stats.txt','w') as f:
            f.write(stats)
        print('Preparing dataset statistics completed!')
