import torch


# Deblur
import models.archs_deblur.eSL_deblur as eSL_deblur
import models.archs_deblur.EVDI as EVDI
import models.archs_deblur.MemDeblur as MemDeblur
import models.archs_deblur.PAN_deblur as PAN_deblur
import models.archs_deblur.STRA1 as STRA1
# import models.archs_deblur.STRAHN1 as STRAHN1
import models.archs_deblur.CCZ as CCZ

# LOL_Blur
import models.archs_deblur.D2HNet as D2HNet
import models.archs_deblur.D2Net as D2Net
import models.archs_deblur.EFNet as EFNet
import models.archs_deblur.ERDN as ERDN
import models.archs_deblur.RED_Net as RED_Net
import models.archs_deblur.STFAN as STFAN
import models.archs_deblur.STRAHN_deblur as STRAHN_deblur
import models.archs_deblur.UEVD as UEVD
import models.archs_deblur.LEDVI as LEDVI


# Derain
import models.archs_derain.JDNet as JDNet
import models.archs_derain.MPRNet as MPRNet
import models.archs_derain.RIDNet as RIDNet
import models.archs_derain.RLNet as RLNet
import models.archs_derain.VRGNet as VRGNet
import models.archs_derain.SPANet as SPANet
import models.archs_derain.DuRN as DuRN
import models.archs_derain.DCSFN as DCSFN

import models.archs_derain.JDNet_event as JDNet_event
import models.archs_derain.MPRNet_event as MPRNet_event
import models.archs_derain.RIDNet_event as RIDNet_event
import models.archs_derain.RLNet_event as RLNet_event
import models.archs_derain.VRGNet_event as VRGNet_event
import models.archs_derain.SPANet_event as SPANet_event
import models.archs_derain.DuRN_event as DuRN_event
import models.archs_derain.DCSFN_event as DCSFN_event
import models.archs_derain.ESTIL as ESTIL
import models.archs_derain.ESTIL_event as ESTIL_event
# import models.archs_derain.STRAHN_visual as STRAHN_visual




# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    ################################### derain
    if which_model == 'JDNet':
        netG = JDNet.JDNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'MPRNet':
        netG = MPRNet.MPRNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'RIDNet':
        netG = RIDNet.RIDNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'RLNet':
        netG = RLNet.RLNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'VRGNet':
        netG = VRGNet.VRGNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'DCSFN':
        netG = DCSFN.DCSFN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'SPANet':
        netG = SPANet.SPANet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'DuRN':
        netG = DuRN.DuRN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'ESTIL':
        netG = ESTIL.ESTIL()

    elif which_model == 'JDNet_event':
        netG = JDNet_event.JDNet_event(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'MPRNet_event':
        netG = MPRNet_event.MPRNet_event(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'RIDNet_event':
        netG = RIDNet_event.RIDNet_event(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'RLNet_event':
        netG = RLNet_event.RLNet_event(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'VRGNet_event':
        netG = VRGNet_event.VRGNet_event(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'DCSFN_event':
        netG = DCSFN_event.DCSFN_event(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'SPANet_event':
        netG = SPANet_event.SPANet_event(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'DuRN_event':
        netG = DuRN_event.DuRN_event(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'ESTIL_event':
        netG = ESTIL_event.ESTIL_event()


    ################################### deblur
    elif which_model == 'eSL_Net_deblur':
        netG = eSL_deblur.eSL_Net_deblur(scale=opt_net['scale'])
    elif which_model == 'STRA1':
        netG = STRA1.STRA1(num_res=opt_net['num_res'])
    elif which_model == 'EVDI':
        netG = EVDI.EVDI(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'MemDeblur':
        netG = MemDeblur.MemDeblur(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'PAN_deblur':
        netG = PAN_deblur.PAN_deblur(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
#     elif which_model == 'STRANH_visual':
#         netG = STRAHN_visual.STRAHN_visual(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
#     elif which_model == 'STRANH1':
#         netG = STRAHN1.STRAHN1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
        ##################################
    elif which_model == 'D2HNet':
        netG = D2HNet.D2HNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])
    elif which_model == 'D2Net':
        netG = D2Net.D2Net(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'EFNet':
        netG = EFNet.EFNet(in_nc=opt_net['in_nc'])
    elif which_model == 'ERDN':
        netG = ERDN.ERDN(in_channels=opt_net['in_nc'])
    elif which_model == 'RED_Net':
        netG = RED_Net.RED_Net(in_nc=opt_net['in_nc'])
    elif which_model == 'STFAN':
        netG = STFAN.STFAN_Net(input_channel=opt_net['in_nc'])
    elif which_model == 'STRAHN_deblur':
        netG = STRAHN_deblur.STRAHN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'UEVD':
        netG = UEVD.UEVD(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'LEDVI':
        netG = LEDVI.LEDVI(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
        
        
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


