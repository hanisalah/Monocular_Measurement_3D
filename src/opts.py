import os
import sys
import torch

class opts(object):
    def __init__(self,paths,settings):

        #locker sizes in meters
        self.locker_sizes = settings['locker_sizes']

        # basic experiment setting
        #settings
        self.exp_id = paths['save_dir'].split('/')[-2]+'/'#'default/'
        self.test = False
        self.debug = int(0)
        self.resume = settings['is_resume']#False#True

        # paths
        self.root_path = paths['root_path']#'/'.join(os.path.abspath('__file__').split('\\')[:-2])+'/data'
        self.data_dir = paths['data_dir'] #self.root_path + '/kitti/'
        self.image_dir = paths['image_dir'] #self.data_dir + 'data_object_image_2/training/image_2/'
        self.calib_dir = paths['calib_dir'] #self.data_dir + 'data_object_calib/training/calib/'
        self.label_dir = paths['label_dir'] #self.data_dir + 'data_object_label_2/training/label_2/'
        self.working_dir = paths['working_dir'] #self.data_dir + 'working/anns/'
        self.demo = paths['demo'] #self.root_path + '/kitti/data_object_image_2/training/demo/'

        self.exp_dir = paths['exp_dir'] #'./exp/'
        self.save_dir = paths['save_dir'] #os.path.join(self.exp_dir,self.exp_id)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.debug_dir = paths['debug_dir'] #os.path.join(self.exp_dir,'debug')
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        self.results_dir = paths['results_dir'] #os.path.join(self.exp_dir,'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.load_model = settings['load_model']#''#self.save_dir + 'model_dla34_1.pth'
        #if self.resume:
            #model_path = paths['model_dir'] #self.save_dir[:-4] if self.save_dir.endswith('TEST') else self.save_dir
            #self.load_model = paths['model_dir'] + settings['load_model']#os.path.join(model_path, 'model_last.pth')

        self.vis = True
        self.dataset = 'custom'#'RTM3D'
        self.coor_thresh = float(0.3) if self.dataset=='kitti' else float(1)

        # system coor_thresh
        self.gpus = [int(0)] if torch.cuda.is_available() else [int(-1)] #[int(0)]
        self.device = int(0) if torch.cuda.is_available() else int(-1) #int(0)
        self.num_workers = int(1) if self.debug >0 else int(4)
        self.not_cuda_benchmark = False
        self.not_cuda_enabled = False
        self.seed = int(317)

        # log
        self.print_iter = int(500)#int(1)
        self.hide_data_time = True
        self.save_all = True
        self.metric = 'loss'
        self.vis_thresh = float(0.3)
        self.debugger_theme = 'white'

        # model
        self.arch = 'dla_34'
        self.head_conv = int(256)
        self.down_ratio = int(4)
        self.pad = int(31)
        self.num_stacks = int(1)

        # input
        self.input_res = int(-1)
        self.input_h = int(-1)
        self.input_w = int(-1)

        # train
        self.lr = float(1.25e-4)
        self.lr_step = [90,120]
        self.num_epochs = int(140)
        self.batch_size = settings['batch_size'] #int(2) # should be int(32) but int(2) is used due to hardware limitation
        self.num_iters = int(-1)
        self.val_intervals = int(5)

        # test
        self.flip_test = False #True
        self.test_scales = [1]
        self.nms = True
        self.K = int(100)
        self.not_prefetch_test = True
        self.fix_res = False
        self.keep_res = True

        # dataset
        self.not_rand_crop = False#True
        self.shift = float(0.1)
        self.scale = float(0.4)
        self.rotate = float(0.0)
        self.flip = float(0.5)
        self.no_color_aug = True
        self.aug_rot = float(0.0)
        self.aug_ddd = float(0.5)
        self.rect_mask = True

        # loss
        self.mse_loss = False #True
        # ctdet
        self.reg_loss = 'l1'
        self.hm_weight = float(1.0)
        self.off_weight = float(1.0)
        self.wh_weight = float(0.1)
        # multi_pose
        self.hp_weight = float(1.0)
        self.hm_hp_weight = float(1.0)
        # ddd
        self.prob_weight = float(1.0)
        self.dim_weight = float(2.0)
        self.rot_weight = float(0.2)
        self.peak_thresh = float(0.2)

        # task
        # ctdet
        self.norm_wh = True
        self.dense_wh = True
        self.cat_spec_wh = True
        self.not_reg_offset = True
        self.reg_offset = not self.not_reg_offset
        # exdet
        self.agnostic_ex = True
        self.scores_thresh = float(0.1)
        self.center_thresh = float(0.1)
        self.aggr_weight = float(0.0)
        # multi_pose
        self.dense_hp = False#True
        self.not_hm_hp = False #True
        self.hm_hp = not self.not_hm_hp
        self.not_reg_hp_offset = False #True
        self.reg_hp_offset = (not self.not_reg_hp_offset) and (not self.not_hm_hp)
        self.not_reg_bbox = False #True
        self.reg_bbox = not self.not_reg_bbox

        #Initializing dataset and dataset statistics
        dataset = self.init(self.dataset, settings)
        self.dataset = dataset.dataset
        self.num_joints = dataset.num_joints
        input_h, input_w = dataset.default_resolution
        self.mean = dataset.mean
        self.std = dataset.std
        self.num_classes = dataset.num_classes
        self.input_h = self.input_h if self.input_h > 0 else input_h
        self.input_w = self.input_w if self.input_w > 0 else input_w
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio
        self.input_res = max(self.input_h, self.input_w)
        self.output_res = max(self.output_h, self.output_w)
        self.det_cats = dataset.det_cats
        self.flip_idx = dataset.flip_idx
        self.heads = {'hm': self.num_classes, 'wh': 2, 'hps': 18,'rot': 8,'dim':3,'prob':1}
        if self.reg_offset:
            self.heads.update({'reg': 2})
        if self.hm_hp:
            self.heads.update({'hm_hp': 9})
        if self.reg_hp_offset:
            self.heads.update({'hp_offset': 2})
        print('heads', self.heads)

        self.help = {
        'locker_sizes': 'list of sizes of lockers in meters. Size is represented by 3 dimensions',

        #basic experiment setting
        'exp_id':'experiment id',
        'debug': 'level of visualization. 1: only show the final detection results, 2: show the network output features, 3: use matplot to display, 4: save all vis to disk',
        'resume': 'resume an experiment. Reloaded optimizer parameter and set load_model to model_last.pth in the exp dir if load_model is empty',
        'root_path': 'root path above project folder',
        'data_dir': 'path to dataset',
        'image_dir': 'path to folder of training images',
        'calib_dir': 'path to folder of calibration files',
        'label_dir': 'path to folder of label files of training images',
        'working_dir': 'path to folder of annotations of dataset',
        'demo': 'path to prediction images',
        'exp_dir': 'path to experiments',
        'save_dir': 'path to save everything',
        'debug_dir': 'path to debug',
        'results_dir': 'path to results',
        'load_model': 'path to pretrained model',
        'vis': '',
        'dataset': 'dataset name',
        'coor_thresh': '',

        # system coor_thresh
        'gpus': '-1 for CPU, else list of GPUs id',
        'device': 'integer identifier of gpu',
        'not_cuda_benchmark': 'disable when input size is not fixed',
        'not_cuda_enabled': 'disable when input size is not fixed',
        'seed': 'random seed from CornerNet',

        #log
        'print_iter': 'disable progress bar and print to screen',
        'hide_data_time': 'not display time during training',
        'save_all': 'save model to disk every 5 epochs',
        'metric': 'main metric to save best model',
        'vis_thresh': 'visualization threshold',
        'debugger_theme': 'white or black',

        #model
        'arch':'model architecture name',
        'head_conv': '0: no conv layer, -1 default setting, 64: resnets, 256: dla',
        'down_ratio': 'output stride. Currently only supports 4.',

        #input
        'input_res': 'input height and width. -1 for default from dataset. Will be overriden by input_h | input_w',
        'input_h': 'input height. -1 for default from dataset',
        'input_w': 'input width. -1 for default from dataset',

        #train
        'lr': 'learning rate for batch size 32',
        'lr_step': 'drop learning rate by 10 for each step in list',
        'num_epochs': 'total training epochs',
        'batch_size': 'batch size',
        'num_iters': 'default: #samples / batch_size',
        'val_intervals': 'number of epochs to run validation',

        #test
        'flip_test': 'flip data augmentation',
        'test_scales': 'multi scale test augmentation',
        'nms': 'run nms in testing',
        'K': 'max number of output objects',
        'not_prefetch_test': 'not use parallel data pre-processing',
        'fix_res': 'fix testing resolution or keep original resolution',
        'keep_res': 'keep the original resolution during validation',

        #dataset
        'not_rand_crop': 'not use the random crop data augmentation from CornerNet',
        'shift': 'when not using random crop apply shift augmentation',
        'scale': 'when not using random crop apply scale augmentation',
        'rotate': 'when not using random crop apply rotation augmentation',
        'flip': 'probability of applying flip augmentation',
        'no_color_aug': 'not use color augmentation from CornerNet',
        'aug_rot': 'probability of applying crop augmentation',
        'aug_ddd': 'probability of applying crop augmentation',
        'rect_mask': 'for ignored object, apply mask on the rectangular region or just center point',

        #loss
        'mse_loss': 'use mseloss or focalloss to train keypoint heatmaps',
        #ctdet
        'reg_loss': 'regression loss: sl1 | l1 | l2',
        'hm_weight': 'loss weight for keypoint heatmaps',
        'off_weight': 'loss weight for keypoint local offsets',
        'wh_weight': 'loss weight for bounding box size',
        #multi_pose
        'hp_weight': 'loss weight for human pose offset',
        'hm_hp_weight': 'loss weight for human keypoint heatmap',
        #ddd
        'prob_weight': 'loss weight for depth',
        'dim_weight': 'loss weight for 3d bounding box size',
        'rot_weight': 'loss weight for orientation',
        'peak_thresh': '',

        #task
        #ctdet
        'norm_wh': '',
        'dense_wh': 'apply weighted regression near center or just apply regression on center point',
        'cat_spec_wh': 'category specific bounding box size',
        'not_reg_offset': 'not regress local offset',
        #exdet
        'agnostic_ex': 'use category agnostic extreme points',
        'scores_thresh': 'threshold for extreme point heatmap',
        'center_thresh': 'threshold for centermap',
        'aggr_weight': 'edge aggregation weight',
        #multi_pose
        'dense_hp': 'apply weighted pose regression near center of just apply regression on centerpoint',
        'not_hm_hp': 'not estimate human joint heatmap, directly use the joint offset from center',
        'not_reg_hp_offset': 'not regress local offset for human joint heatmaps',
        'reg_hp_offset': '',
        'not_reg_bbox': 'not regression bounding box size',

        }

    def init(self,name, settings):
        default_dataset_info = {
            'RTM3D': {
                'default_resolution': [384, 1280], 'num_classes': 3,
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'kitti_hp', 'num_joints': 9,#8
                'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8]],
                'det_cats': ['Car', 'Pedestrian', 'Cyclist']},
            'KM3D': {
                'default_resolution': [384, 1280], 'num_classes': 3,
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'kitti_hp', 'num_joints': 9,#8
                'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8]],
                'det_cats': ['Car', 'Pedestrian', 'Cyclist']},
            'KM3D_nuscenes': {
                'default_resolution': [896, 1600], 'num_classes': 10,
                'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                'dataset': 'nuscenes_hp', 'num_joints': 9,#8
                'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8]],
                'det_cats': ['Car', 'Pedestrian', 'Cyclist']},
            'custom': {
                'default_resolution': [], 'num_classes': 0,
                'mean':'', 'std':'',
                'dataset':'custom', 'num_joints':9,
                'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8]],
                'det_cats': settings['det_cats']},#['Car', 'Pedestrian', 'Cyclist']},
                }
        if name=='custom':
            with open(self.working_dir+'stats.txt','r') as f:
                lines = f.readlines()
            for line in lines:
                l = line[:-1].split(' ')
                if l[0] == 'default_resolution':
                    default_dataset_info['custom']['default_resolution'] = [ int(l[i]) for i in range(1,len(l))]
                if l[0] == 'num_classes':
                    default_dataset_info['custom']['num_classes'] = int(l[1])
                elif l[0] == 'mean':
                    default_dataset_info['custom']['mean'] = [float(l[i]) for i in range(1,len(l))]
                elif l[0] == 'std':
                    default_dataset_info['custom']['std'] = [float(l[i]) for i in range(1,len(l))]

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)
        dataset = Struct(default_dataset_info[name])
        return dataset
