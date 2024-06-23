import torch
import models.archs.PAN_arch as PAN_arch
import models.archs.PAN_event1_arch as PAN_event1_arch
import models.archs.PAN_event2_arch as PAN_event2_arch
import models.archs.PAN_event3_arch as PAN_event3_arch
import models.archs.PAN_event4_arch as PAN_event4_arch
import models.archs.PAN_event5_arch as PAN_event5_arch
import models.archs.eSL as eSL
import models.archs.e2sri as e2sri
import models.archs.dcsr as dcsr
import models.archs.DPT as DPT
import models.archs.spade_e2v as SPADE
import argparse
import options.options as option
import models.archs.TDAN_model as TDAN
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.RCAN_arch as RCAN_arch
import models.archs.EVSR as EVSR
import models.archs.PAN_event5_arch_group as PAN_event5_arch_group
import models.archs.DDBPN as DDBPN


# Deblur
import models.archs_deblur.eSL_deblur as eSL_deblur
import models.archs_deblur.EVDI as EVDI
import models.archs_deblur.MemDeblur as MemDeblur
import models.archs_deblur.PAN_deblur as PAN_deblur
import models.archs_deblur.STRA1 as STRA1
import models.archs_deblur.STRAHN1 as STRAHN1
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



# SR
import models.archs_SR.BSRDM as BSRDM
import models.archs_SR.DPT as DPT
import models.archs_SR.eSL as eSL_SR
import models.archs_SR.LBNet as LBNet
import models.archs_SR.MRVSR as MRVSR
import models.archs_SR.PAN_SR as PAN_SR
import models.archs_SR.SLS as SLS
import models.archs_SR.TTVSR as TTVSR
import models.archs_SR.EVSR_SR as EVSR_SR
import models.archs_SR.PAN_SR as PAN_SR
import models.archs_SR.STRA_SR as STRA_SR
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
import models.archs_derain.STRAHN_visual as STRAHN_visual

# dehaze
import models.archs_dehaze.AODNet as AODNet
import models.archs_dehaze.DCPDN as DCPDN
import models.archs_dehaze.DehazeFlow as DehazeFlow
import models.archs_dehaze.EPDN as EPDN
import models.archs_dehaze.GCANet as GCANet
import models.archs_dehaze.LDP as LDP
import models.archs_dehaze.PGCNet as PGCNet
import models.archs_dehaze.FFANet as FFA
import models.archs_dehaze.GridDehazeNet as GridDehaze
import models.archs_dehaze.MSBDN as MSBDN
import models.archs_dehaze.PAN_dehaze as PAN_dehaze


# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    ############################### SR
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
    elif which_model == 'DDBPN':
        netG = DDBPN.DDBPN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'eSL':
        netG = eSL.eSL_Net(scale=opt_net['scale'])
    elif which_model == 'e2sri':
        netG = e2sri.SRNet(scale=opt_net['scale'])
    elif which_model == 'DCSR':
        netG = dcsr.DCSR(scale = opt_net['scale'],n_feats=opt_net['n_feats'])
    elif which_model == 'TDAN':
        netG = TDAN.TDAN_VSR()
    elif which_model == 'DPT':
        netG = DPT.DPT_Net(angRes=5,factor=opt_net['scale'])
    elif which_model == 'SPADE':
        netG = SPADE.Unet6(scale=opt_net['scale'])
    elif which_model == 'MSRResNet_PA':
        netG = SRResNet_arch.MSRResNet_PA(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RCAN_PA':
        netG = RCAN_arch.RCAN_PA(n_resgroups=opt_net['n_resgroups'], n_resblocks=opt_net['n_resblocks'], n_feats=opt_net['n_feats'], res_scale=opt_net['res_scale'], n_colors=opt_net['n_colors'], rgb_range=opt_net['rgb_range'], scale=opt_net['scale'])    
    elif which_model == 'EVSR':
        netG = EVSR.EVSR(in_nc=3,out_nc=3,nf=40,unf=24,nb=16,scale=opt_net['scale'])

    ################################### SR
    elif which_model == 'BSRDM':
        netG = BSRDM.BSRDM(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'eSL_SR':
        netG = eSL_SR.eSL_Net_SR(scale=opt_net['scale'])
    elif which_model == 'LBNet':
        netG = LBNet.LBNet(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'MRVSR':
        netG = MRVSR.MRVSR(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'SLS':
        netG = SLS.SLS(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'STRA_SR':
        netG = STRA_SR.STRA_SR(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])

    elif which_model == 'TTVSR':
        netG = TTVSR.TTVSR(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])

    elif which_model == 'EVSR_SR':
        netG = EVSR_SR.EVSR_SR(in_nc=opt_net['in_nc'],out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'],nb=opt_net['nb'],scale=opt_net['scale'])
    elif which_model == 'PAN_SR':
        netG = PAN_SR.PAN_Event_3(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])



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
    elif which_model == 'STRANH1':
        netG = STRAHN1.STRAHN1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'STRANH_visual':
        netG = STRAHN_visual.STRAHN_visual(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
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

    ################################### derain
    elif which_model == 'JDNet':
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
    
    
    ################################### dehaze
    elif which_model == 'AODnet':
        netG = AODNet.AODnet(in_nc=opt_net['in_nc'])
    elif which_model == 'DCPDN':
        netG = DCPDN.Dense()
    elif which_model == 'DehazeFlow':
        netG = DehazeFlow.DehazeFlow(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['scale'])
    elif which_model == 'EPDN':
        netG = EPDN.EPDN(in_nc=opt_net['in_nc'])
    elif which_model == 'FFA':
        netG = FFA.FFA(in_nc=opt_net['in_nc'],gps=3,blocks=19)
    elif which_model == 'GCANet':
        netG = GCANet.GCANet(in_c=opt_net['in_nc'],out_c=opt_net['out_nc'])
    elif which_model == 'GridDehazeNet':
        netG = GridDehaze.GridDehazeNet(in_channels=opt_net['in_nc'])
    elif which_model == 'LDP':
        netG = LDP.LDP(input_nc=opt_net['in_nc'],output_nc=opt_net['out_nc'])
    elif which_model == 'PAN_dehaze':
        netG = PAN_dehaze.PAN_dehaze(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],nf=opt_net['nf'],unf=opt_net['unf'], nb=opt_net['nb'], scale=opt_net['scale'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


