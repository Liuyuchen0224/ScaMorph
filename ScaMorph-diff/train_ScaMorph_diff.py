from torch.utils.tensorboard import SummaryWriter
import os, utils as utils, glob, losses, sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from model import ScaMorph_diff
import argparse
import time 
import torch.nn as nn

class Logger(object):
    def __init__(self, save_dir):
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        self.terminal = sys.stdout
        self.loglist = natsorted(glob.glob(save_dir+'*.log'))
        while len(self.loglist) > 5:
            os.remove(self.loglist[0])
            self.loglist = natsorted(glob.glob(save_dir + '*.log'))
        self.log = open(save_dir+f"log_{self.time}.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
parser = argparse.ArgumentParser()
# data organization parameters
parser.add_argument('--atlas_dir', default='dir/IXI/atlas.pkl', help='atlas_dir files')
parser.add_argument('--train_dir', default='dir/IXI/Train/',help='Train_dir path')
parser.add_argument('--val_dir', default='dir/IXI/Val/',help='Val_dir path')
parser.add_argument('--model_name', default='ScaMorphDiff',help='model output directory (default: models)')

# training parameters
parser.add_argument('--img_size', default=(160, 192, 224), help='Img Size')
parser.add_argument('--gpu', default=0, help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--load_weights', help='optional model file to initialize with')
parser.add_argument('--initial_epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')



# loss hyperparameters
parser.add_argument('--image_loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--sim_lambda', type=float, default=1,
                    help='weight of sim loss (default: 10)')
parser.add_argument('--kl_lambda', type=float, default=1,
                    help='prior lambda regularization for KL loss (default: 10)')
parser.add_argument('--reg_lambda', type=float, default=1,
                    help='weight of gradient or KL loss (default: 0.01)')
args = parser.parse_args()


def main():
        
    save_dir = '{}_{}{}_reg{}/'.format(args.model_name,args.image_loss,args.sim_lambda, args.reg_lambda)
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    print(f'Write logs into logs/{save_dir}')

    '''
    Initialize model
    '''
    model = ScaMorph_diff(args.img_size)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(args.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(args.img_size, 'bilinear')
    reg_model_bilin.cuda()


    '''
    If continue from previous training
    '''
    if args.load_weights:
        model_dir = 'experiments/'+save_dir
        load_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])
        best_model = load_model['state_dict']
        epoch_start = load_model['epoch']
        updated_lr = round(args.lr * np.power(1 - (epoch_start) / args.epochs,0.9),8)
        model.load_state_dict(best_model)
        print('Model Load DONE. Begin train at epoch',epoch_start)
    else:
        epoch_start = args.initial_epoch
        updated_lr = args.lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])

    val_composed = transforms.Compose([trans.Seg_norm(),
                                       trans.NumpyType((np.float32, np.int16)),
                                        ])
    train_set = datasets.IXIBrainDataset(glob.glob(args.train_dir + '*.pkl'), args.atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(args.val_dir + '*.pkl'), args.atlas_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
 
    # prepare image loss
    if args.image_loss == 'ncc':
        criterion_sim = losses.NCC().loss
        weights = [args.sim_lambda]
    elif args.image_loss == 'mse':
        criterion_sim = losses.MSE().loss
        weights = [args.sim_lambda]
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)
    
    criterion_reg = losses.Grad3d(penalty='l2')
    weights += [args.reg_lambda]
    
    criterion_kl = losses.VM_diffeo_loss(image_sigma=0.02, prior_lambda=10, flow_vol_shape=args.img_size).cuda()
    weights += [args.kl_lambda]    
    best_dsc = 0
    
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    print('Training Starts')
    for epoch in range(epoch_start, args.epochs):

        '''
        Training
        '''
        epoch_step_time = utils.AverageMeter()
        sim_loss = utils.AverageMeter()
        reg_loss = utils.AverageMeter()
        kl_loss = utils.AverageMeter()
        epoch_total_loss = utils.AverageMeter()
        epoch_start_time = time.time()
        
        idx = 0
        for data in train_loader:
            step_start_time = time.time()
            idx += 1
            
            model.train()
            data = [t.cuda() for t in data]
            
            x = data[0]
            y = data[1]
        
            x_in = torch.cat((x,y), dim=1)
            output,flow_param,flow = model(x_in)
            
            loss_sim = criterion_sim(output, y)
            loss_reg = criterion_reg(flow, y)
            loss_kl = criterion_kl.kl_loss(flow_param)
                         
            loss = loss_sim * weights[0] +  loss_reg* weights[1] + loss_kl * weights[2]
                
            sim_loss.update(loss_sim.item(), y.numel())           
            reg_loss.update(loss_reg.item(), y.numel())
            kl_loss.update(loss_kl.item(), y.numel())           
            epoch_total_loss.update(loss.item(), y.numel())    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
            epoch_step_time.update(time.time() - step_start_time, y.numel())

            sys.stdout.write("\r" + 'Epoch {}/{} idx {}: Total_loss: {:.4f} Sim: {:.6f} Reg: {:.6f} KL:{:.6f} {:.4f} sec/step '.format(epoch,args.epochs,idx, epoch_total_loss.val, sim_loss.val, reg_loss.val,kl_loss.val,epoch_step_time.val))
            sys.stdout.flush()            
            del output
            torch.cuda.empty_cache()
        # writer.add_scalar('Loss/train', loss_list.avg, epoch)
        epoch_time=time.time() - epoch_start_time
        writer.add_scalar('Loss/train', epoch_total_loss.avg, epoch)
        sys.stdout.write('Epoch {} loss {:.4f} time {:.4f}'.format(epoch, epoch_total_loss.avg,epoch_time))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                grid_img = mk_grid_img(8, 1, args.img_size)
                _,_, flow = model(torch.cat((x,y),dim=1))
                
                def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/'+save_dir, pre='dsc',filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        print('Save checkpoint done... Checkpoint name: dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        if eval_dsc.avg > best_dsc:        
            best_dsc = eval_dsc.avg
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, save_dir='experiments/'+save_dir, pre='best',filename='best_dice.pth.tar'.format(eval_dsc.avg))
            print('NEW Best Dice!!! Save checkpoint done... Checkpoint name:best_dice.pth.tar')
              
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', pre='dsc',filename='checkpoint.pth.tar', max_model_num=2):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + f'{pre}*.pth.tar'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + f'{pre}*.pth.tar'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    utils.fix_seed(2023)
    GPU_iden = args.gpu
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    assert GPU_avai, 'No CUDA found!'
    print('If the GPU is available? ' + str(GPU_avai))
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    main()