from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd, build_ssd_efficientnet
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import pickle
import math


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch') # SSD argument parsing 선언.
train_set = parser.add_mutually_exclusive_group() # 상호 배타적인 그룹을 만든다. argparse 는 상호 배타적인 그룹에서 오직 하나의 인자만 명령행에 존재하는지 확인한다.

# argument parsing으로 train.py를 실행할 때 parameter들을 변경해주면서 실행할 수 있다. 반드시 모든 add_argument를 실행할 때 적용해야하는 것은 아니며 바꿀 필요가 있는
# parameter에 대해서만 실행하면 된다. 나머지는 default값으로 적용되기 때문.
parser.add_argument('--input',default=512, type=int, choices=[300, 512], help='ssd input size, currently support ssd300 and ssd512') # 어떤 SSD를 사용할지 결정 300 or 512
parser.add_argument('--dataset', default='VOC',
                    type=str, help='VOC or COCO') # dataset 결정. VOC or COCO
parser.add_argument('--num_class', default=6, type=int, help='number of class in ur dataset') # 내가 적용 시킬 dataset class 개수 입력. (custom data일때)
parser.add_argument('--dataset_root', default='/content/drive/MyDrive/SSD.Pytorch/kitti/kitti',
                    help='Dataset root directory path') # custom dataset일때 새로운 dataset 경로를 입력.
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', type=str, choices=['vgg16_reducedfc.pth', 'efficientnet_b4_truncated.pth'],
                    help='Pretrained base model') # backbone이 되는 architecture를 가져온다. vgg16이나 efficientnet 중 결정.
parser.add_argument('--num_epoch', default=300, type=int, help='number of epochs to train') # training 반복할 epoch 횟수.
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training') # 얼마나 묶어서 training할 지 batch size로 정한다.
parser.add_argument('--resume', default='/content/drive/MyDrive/SSD.Pytorch/weights/ssd512_VOC_10000.pth', type=str,
                    help='Checkpoint state_dict file to resume training from') # 학습이 중간에 종료된다거나 오류가 발생했을 때 학습하면서 checkpoint를 정해 가중치를 저장하기 때문에
# 다시 그 가중치들을 사용함으로써 중단되었던 구간부터 다시 resume할 수 있게 하는 명령어.
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Resume training at this epoch') # resume할 때 반복시점을 넣어줄 수 있는 명령어.
parser.add_argument('--num_workers', default=6, type=int,
                    help='Number of workers used in dataloading') # GPU나 CPU에 올려서 학습을 진행하게 되는데 num_workers로 그 작업하는 개수의 단위를 적용한다.
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model') # 작업환경을 GPU로 적용할지 CPU, 또는 다른 기반으로 할 지 선택.
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate') # learning rate는 기본적으로 0.001로 되어있지만 사용자 설정으로 정할 수 있다.
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim') # optimization을 위해서 쓰이는 공식에 들어가는 momentum 계수를 설정할 수 있다.
parser.add_argument('--weight_decay', default=1e-8, type=float,
                    help='Weight decay for SGD') # 가중치 감쇠효과를 줄 수 있는데 이 hyperparameter를 사용자가 설정 가능.
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD') # SGD 기법에 사용되는 gamma 계수를 설정 가능.
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization') # visdom을 사용해서 학습되는 상태를 visualize할지 말지에 대한 부분.
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models') # checkpoint 지점의 weight를 저장할 경로 설정.
args = parser.parse_args() # 입력받은 인자값들을 args에 저장.


if torch.cuda.is_available(): # GPU가 사용가능할 때 사용자 설정에 의해 CPU가 사용될때와 GPU를 사용할 때 출력하는 문구.
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # GPU 사용.
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
        # GPU가 사용가능하다고 출력하는 문구 출력과 동시에 CPU사용.
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    # GPU가 사용불가능이므로 그냥 CPU사용.
    
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
    # weight 저장할 경로 설정.


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
                # COCO dataset 경로가 아니라면 오류를 발생하게 함.
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            # 사용하려는 dataset은 COCO인데 dataset root를 VOC dataset으로 설정하였으므로 오류문구 출력. 하지만 dataset 경로를 COCO로 변경은 해준다.
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
        # dataset 경로를 COCO로 변경하고 불러온다.
        
    elif args.dataset == 'VOC':
        if args.dataset_root == VOC_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
            # 위의 COCO부분과 동일.
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(args.input,
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
        # visdom import한 뒤 실행.
    if args.basenet == 'vgg16_reducedfc.pth':
        ssd_net = build_ssd('train', args.input, args.num_class) # ssd_net을 vgg16 network로 구성.
        # backbone 네트워크 설정에 따라 그 네트워크를 지정.
    elif args.basenet == 'efficientnet_b4_truncated.pth':
        ssd_net = build_ssd_efficientnet('train', args.input, args.num_class) # ssd_net을 efficient network로 구성.
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        # GPU를 사용한다면 데아터 병령처리를 통해 ssd_net 다시 설정.
        cudnn.benchmark = True
        # GPU 성능 테스트 확인

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
        # resume을 원한다면 저장했던 weight경로에서 data 불러오기.
    else:
        if args.basenet == 'vgg16_reducedfc.pth':
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network weights from %s\n'%(args.save_folder + args.basenet))
            ssd_net.base.load_state_dict(vgg_weights)
            # basenet이 vgg16이라면 vgg16의 가중치를 가져오도록 설정.
        elif args.basenet == 'efficientnet_b4_truncated.pth':
            efficientnet_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network weights from %s\n' % (args.save_folder + args.basenet))
            print('ssd_net.base:',ssd_net.base)
            ssd_net.base.load_state_dict(efficientnet_weights)
            # 마찬가지로 efficientnet이 basenet일 때 가중치를 load할 dictionary를 efficientnet으로 설정

    if args.cuda:
        net = net.cuda()
        # GPU 사용시 위에서 구성한 network를 GPU로 올린다.

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method

        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr) 
    # optimization 기법을 Adam을 지정, 그리고 learning rate를 사용자 설정의 learning rate로 적용.
    criterion = MultiBoxLoss(args.num_class, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda) # num class와 확률 값, boolean을 넣어주면서 multi task loss 판별식을 만들어준다.

    net.train()
    # training 시작
    
    loc_loss = 0
    conf_loss = 0
    iteration = 1
    loss_total = []
    loss_loc = []
    loss_cls = []
    # 초기화
    
    print('Loading the dataset...')

    epoch_size = math.ceil(len(dataset) / args.batch_size)
    # math.ceil은 1단위로 올림을 실행한다. dataset을 batchsize로 얼마나 나누냐에 따라 결정.
    print('iteration per epoch:',epoch_size)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)
    step_index = 0
    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
        # visdom을 통해서 visualize를 할 목록을 세팅.

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # data를 training할 때 업로드(적용)할 때 batchsize와 num_workers로 얼마나씩 적용할지 설정. -> batch iterator
    
    for epoch in range(args.start_epoch, args.num_epoch):
        print('\n'+'-'*70+'Epoch: {}'.format(epoch)+'-'*70+'\n')
        if args.visdom and epoch != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # visdom이 설정되었고 epoch이 0이 아니라면 epochsize 단위마다 visdom을 업데이트 시키면서 visualize가동.
            
            loc_loss = 0
            conf_loss = 0
            epoch += 1
            # 표시할 loss값들을 다음을 위해 초기화.
        if epoch in cfg['SSD{}'.format(args.input)]['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            # step size에 따라서 learning rate를 감쇠시키는 목적.
        if epoch <= 5:
            warmup_learning_rate(optimizer,epoch)
            # 초기 epoch일 때 learning rate 설정.
        for images, targets in data_loader: # train data 로딩.
            for param in optimizer.param_groups:
                if 'lr' in param.keys():
                    cur_lr = param['lr']
                    # learning rate 가져오기.
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]
            # images를 저장하기 위해 GPU 사용 유무에 따라 먼저 images를 랜덤으로 초기화.
            
            t0 = time.time() # 이 때의 시간 설정.
            out = net(images) # images를 network에 넣어서 예측값 out 출력.
            
            optimizer.zero_grad() # 처음으로 gradient 0으로 설정.
            loss_l, loss_c = criterion(out, targets) # 아까 만든 판별식으로 loss값 도출.
            loss = loss_l + loss_c # multitask에 의한 총 loss 계산.
            loss.backward() # loss를 적용해서 backwardpropogation 실행.
            optimizer.step() # Adam optimization을 통해 loss에 따라 가중치를 업데이트.
            t1 = time.time() # 이 때의 시간 설정.
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            # 각 loss값 저장.

            if iteration % 10 == 0:
                print('Epoch '+repr(epoch)+'|| iter ' + repr(iteration % epoch_size)+'/'+repr(epoch_size) +'|| Total iter '+repr(iteration)+ ' || Total Loss: %.4f || Loc Loss: %.4f || Cls Loss: %.4f || LR: %f || timer: %.4f sec.\n' % (loss.item(),loss_l.item(),loss_c.item(),cur_lr,(t1 - t0)), end=' ')
                loss_cls.append(loss_c.item())
                loss_loc.append(loss_l.item())
                loss_total.append(loss.item())
                loss_dic = {'loss':loss_total, 'loss_cls':loss_cls, 'loss_loc':loss_loc}
                # iteration 10의 배수일때마다 loss 저장.

            if args.visdom:
                update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                                iter_plot, epoch_plot, 'append')
                # visualize위해 visdom plot 업데이트.

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), 'weights/ssd{}_VOC_'.format(args.input) +
                           repr(iteration) + '.pth')
                # iteration이 5000번 단위일때마다 weight를 저장.
                with open('loss.pkl', 'wb') as f:
                    pickle.dump(loss_dic, f, pickle.HIGHEST_PROTOCOL)
                    # pickle을 이용해 데이터 저장.
            iteration += 1
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')
    # 현재 상태 저장.

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step)) # 가중치 감쇠.
    print('Now we change lr ...')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # 가중치 감쇠효과 적용하는 함수.

def warmup_learning_rate(optimizer,epoch):
    lr_ini = 0.0001
    print('lr warmup...')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_ini+(args.lr - lr_ini)*epoch/5
    # 초기 learning rate를 약간 높여줘서 빠르게 learning 될 수 있게 함.

def xavier(param):
    init.xavier_uniform_(param)
    # xavier 초기화 방법 사용시.
    

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    # 재비어 초기화 방법으로 가중치 초기화, 바이어스 0으로 초기화.


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )
# visdom의 plot을 만드는 함수. 
# cpu를 사용해 torch로 좌표를 만들고 title과 label 등 설정.

def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # visdom plot에 iteration에 따라서 update된 loss를 갱신한다.

    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )
    # iteration이 0이라면 다시 초기화해서 빈 plot으로 만듬.    

if __name__ == '__main__':
    train()
    # train 가동.
