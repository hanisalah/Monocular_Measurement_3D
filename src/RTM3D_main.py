import os
import torch
import torch.utils.data
import glob
from copy import deepcopy
from src.opts import opts
from src.model import create_model, load_model, save_model, BaseTrainer, BaseDetector
from src.utils import Logger
from src.dataset import DSet

def train(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark
    torch.backends.cudnn.enabled = not opt.not_cuda_enabled
    logger = Logger(opt)

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = BaseTrainer
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(DSet(opt,'val'),batch_size=1, shuffle=False, num_workers=1,pin_memory=True)

    train_loader = torch.utils.data.DataLoader(DSet(opt, 'train'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print('epoch: ',epoch)
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(opt.save_dir + 'model_{}.pth'.format(mark), epoch, model, optimizer) #(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best and epoch != 1:
                best = log_dict_val[opt.metric]
                save_model(opt.save_dir+'model_best.pth', epoch, model) #(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
        else:
            save_model(opt.save_dir+'model_last.pth', epoch, model, optimizer) #(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(opt.save_dir + 'model_{}.pth'.format(epoch), epoch, model, optimizer) #(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                  param_group['lr'] = lr
    logger.close()

def predict(opt,img_format='.png'):
    opt.debug = max(opt.debug, 1)
    opt.faster=False
    Detector = BaseDetector
    detector = Detector(opt)
    print('results_dir',opt.results_dir)
    files = sorted(glob.glob(opt.demo+'*'+img_format))
    image_names = [f.replace('\\','/') for f in files]
    for (image_name) in image_names:
        print(image_name)
        ret = detector.run(image_name)

#Added
def measure(opt, image, calib_master_path):
    result = {}
    opt.faster=False
    Detector = BaseDetector
    detector = Detector(opt)
    #if image!='' and calib_master_path!='':
    result = detector.run(image, calib_master_path)
    return deepcopy(result)


def runny(mode, paths, settings, image='', calib_master_path=''):#img_format='.png'):
    opt = opts(paths, settings)
    if mode =='train':
        train(opt)
        return {}
    elif mode == 'predict':
        #predict(opt,img_format=img_format)
        result = measure(opt,image,calib_master_path)
        return deepcopy(result)
