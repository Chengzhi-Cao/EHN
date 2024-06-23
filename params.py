import torch
import models.archs.PAN_arch as PAN_arch
import models.archs.PAN_event1_arch as PAN_event1_arch
import models.archs.PAN_event2_arch as PAN_event2_arch
import models.archs.PAN_event3_arch as PAN_event3_arch
import models.archs.PAN_event4_arch as PAN_event4_arch
import models.archs.eSL as eSL
import models.archs.e2sri as e2sri
import models.archs.dcsr as dcsr
import models.archs.DPT as DPT
import argparse
import options.options as option
import models.archs.TDAN_model as TDAN
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.spade_e2v as SPADE
                                                    
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'PAN':
        netG = PAN_arch.PAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])

    elif which_model == 'PAN_Event1':
        netG = PAN_event1_arch.PAN_Event_1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'PAN_Event2':
        netG = PAN_event2_arch.PAN_Event_2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'PAN_Event3':
        netG = PAN_event3_arch.PAN_Event_3(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'PAN_Event4':
        netG = PAN_event4_arch.PAN_Event_4(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'eSL':
        netG = eSL.eSL_Net(scale=opt_net['scale'])
    elif which_model == 'e2sri':
        netG = e2sri.SRNet(scale=opt_net['scale'],base1_channels=opt_net['base1_channels'],base2_channels=opt_net['base2_channels'])
    elif which_model == 'DCSR':
        netG = dcsr.DCSR(scale = opt_net['scale'],n_feats=opt_net['n_feats'])
    elif which_model == 'TDAN':
        netG = TDAN.TDAN_VSR()
    elif which_model == 'DPT':
        netG = DPT.DPT_Net(angRes=5,factor=opt_net['scale'])


    elif which_model == 'MSRResNet_PA':
        netG = SRResNet_arch.MSRResNet_PA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RCAN_PA':
        netG = RCAN_arch.RCAN_PA(n_resgroups=opt_net['n_resgroups'], n_resblocks=opt_net['n_resblocks'], n_feats=opt_net['n_feats'], res_scale=opt_net['res_scale'], n_colors=opt_net['n_colors'], rgb_range=opt_net['rgb_range'], scale=opt_net['scale'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str,default='options/train/train_e2sri_Eventx2_EvInt_param7e5.yml', help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    opt_net = opt['network_G']
    # which_model = opt_net['which_model_G']

    model = PAN_arch.PAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    from torchstat import stat
    # model = DPT.DPT_Net(angRes=5,factor=opt_net['scale'])

    stat(model, (3,40,40))