import os
import yaml
import argparse
import numpy as np
import imageio

import torch
import torchio as tio
import nibabel as nib
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from utils.data_pipeline import InferenceReader, overlap_labels, discretize_labels, ETThresholdSuppression, RemoveMinorConnectedComponents
from utils.iterator import MetricMeter
from models.networks import PriorAttentionNet, UNet, UNet_L4, UNet_L4_LKDW, UNet_L4_Deform, UNet_L4_ASPMLP, UNet_L4_GateCASPMLP, \
    UNet_L4_Gate_CASPMLP, UNet_L4_Gate_CASPMLP_B, \
    UNet_L4_PoolFormer, UNet_L4_ConvFormer, UNet_L4_ConvAttFormer, \
    UNet_L4_Former_CSMLP, UNet_L4_Former_CSHMLP, UNet_L4_ConvFormer_CSMLP, UNet_L4_Former_CSMLP_LKDW, UNet_L4_Former_CSMLP_LKDW_Deform,\
    UNet_LGSM_MCA_Deform, UNet_LGSM_MCA_MSADC, UNet_LGSM_MCA_MSADC_Plus, \
    Baseline, UNet_MCA, UNet_MSADC, \
    EnhancedUNet, AttentionUNet, CascadedUNet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="BraTS2019", help='dataset name')
    ## Chose run device
    On_Device = '3090'  # A4000 | 3090 | 3090_Dell
    Is_Visual = False
    if On_Device == '3090_Dell':
        if Is_Visual:
            parser.add_argument('--root', default="/home/ff/jy/datasets/BraTS2019/MICCAI_BraTS_2019_Data_Validation_Visual",  # _Visual
                                help='validation dataset path')
        else:
            # parser.add_argument('--root', default="/home/ff/jy/datasets/BraTS2019/MICCAI_BraTS_2019_Data_Validation",
            #                     help='train dataset path')
            parser.add_argument('--root', default="/home/ff/jy/datasets/BraTS2019/MICCAI_BraTS2019_ValidationData",
                                help='validation dataset path')
    elif On_Device == '3090':
        if Is_Visual:
            parser.add_argument('--root', default="/home/ff/dataset/jy/BraTS2019/MICCAI_BraTS_2019_Data_Validation_Visual",  # _Visual
                                help='validation dataset path')
        else:
            # parser.add_argument('--root', default="/datasets/jy/BraTS2019/MICCAI_BraTS_2019_Data_Validation",
            #                     help='train dataset path')
            parser.add_argument('--root', default="/home/ff/dataset/jy/BraTS2019/MICCAI_BraTS_2019_Data_Validation",  # 2020 MICCAI_BraTS2019_ValidationData
                                help='validation dataset path')
    else:
        raise Exception('[error]Device can not fond!')

    parser.add_argument('--model', type=str, default='fuse',
                        help='network for training, support UNet and FuseUNet')
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='model checkpoint used for breakpoint continuation')
    parser.add_argument('--gpu', type=str, default='0', help='GPU No')
    parser.add_argument('--patch_test', action="store_true",
                        help='whether to use patched-pipeline when inference')
    parser.add_argument('--validation', action="store_true",
                        help='whether to run inference on BraTS Validation Dataset')
    parser.add_argument('--train', action="store_true",
                        help='whether to run inference on BraTS Training Dataset')
    parser.add_argument('--tta', action="store_true", help='whether to use TTA when inference')
    parser.add_argument('--visual', action="store_true",
                        help='whether to generate visual when inference')
    parser.add_argument('--snapshot', action="store_true",
                        help='whether to generate snapshot when inference')
    return parser.parse_args()


def main():
    # affine array defines the position of the image array data in a reference space
    affine_array = np.array([[-1, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]])

    args = parse_args()
    model_dict = {'unet': UNet,
                  'unet_l4': UNet_L4,
                  'unet_l4_lkdw': UNet_L4_LKDW,
                  'unet_l4_deform': UNet_L4_Deform,
                  'unet_l4_aspmlp': UNet_L4_ASPMLP,
                  'unet_l4_poolformer': UNet_L4_PoolFormer,
                  'unet_l4_convformer': UNet_L4_ConvFormer,
                  'unet_l4_convattformer': UNet_L4_ConvAttFormer,
                  'unet_l4_gate_caspmlp': UNet_L4_Gate_CASPMLP,
                  'unet_l4_gate_caspmlp_b': UNet_L4_Gate_CASPMLP_B,
                  'unet_l4_former_csmlp': UNet_L4_Former_CSMLP,
                  'unet_l4_former_csmlp_lkdw': UNet_L4_Former_CSMLP_LKDW,
                  'unet_l4_former_csmlp_lkdw_deform': UNet_L4_Former_CSMLP_LKDW_Deform,
                  'unet_lgsm_mca_deform': UNet_LGSM_MCA_Deform,
                  'unet_lgsm_mca_msadc': UNet_LGSM_MCA_MSADC,
                  'unet_lgsm_mca_msadc_plus': UNet_LGSM_MCA_MSADC_Plus,
                  'unet_l4_convformer_csmlp': UNet_L4_ConvFormer_CSMLP,
                  'unet_l4_former_cshmlp': UNet_L4_Former_CSHMLP,
                  'baseline': Baseline,
                  'unet_mca': UNet_MCA,
                  'unet_msadc': UNet_MSADC,
                  's2ca_net': UNet_LGSM_MCA_MSADC_Plus,
                  'panet': PriorAttentionNet,
                  'attention_unet': AttentionUNet,
                  'enhanced': EnhancedUNet,
                  'cascade': CascadedUNet
                  }

    model_name = args.model
    dataset_name = args.dataset
    gpu_no = args.gpu
    patch_test = args.patch_test
    use_validation_set = args.validation
    use_trainset = args.train
    checkpoint_path = args.checkpoint
    is_visual = args.visual
    assert model_name in model_dict.keys(), 'Model name not found!'

    with open('configs/adam.cfg', 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)
    if is_visual:
        pred_savedir = './predictions/{}/{}/visual'.format(dataset_name, model_name)
    else:
        pred_savedir = './predictions/{}/{}/submission'.format(dataset_name, model_name)
    os.makedirs(pred_savedir, exist_ok=True)

    dataseed = cfg['TRAIN']['DATASEED']
    patch_size = cfg['TRAIN']['PATCHSIZE']  # size of each patch, works for patched inference
    patch_overlap = int(cfg['TEST']['PATCHOVERLAP'] * patch_size)  # overlap of patches
    optimize_overlap = cfg['TRAIN']['OVERLAP']  # whether to optimize directly on overlap regions (et, tc and wt)
    use_leaky = cfg['MODEL']['LEAKY']
    norm_type = cfg['MODEL']['NORM']
    use_deconv = cfg['MODEL']['DECONV']  # use deconv or interpolation for upsampling
    suppress_thr = cfg['TEST']['THR']  # suppress threshold for enhancing tumor suppression
    is_global = cfg['TEST']['GLOBAL']  # global replacement of et

    device = torch.device(f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')

    # define network
    if model_name in ['unet', 'attention_unet', 'panet']:
        # model = model_dict[model_name](num_classes=4, input_channels=4, use_deconv=use_deconv,
        #                                channels=(32, 64, 128, 256, 320), strides=(1, 2, 2, 2, 2),
        #                                leaky=use_leaky, norm=norm_type).to(device).eval()
        model = model_dict[model_name](num_classes=4, input_channels=4, use_deconv=use_deconv,
                                       leaky=use_leaky, norm=norm_type).to(device).eval()
    else:  # for l4 model
        model = model_dict[model_name](num_classes=4, input_channels=4, use_deconv=use_deconv,
                                       channels=(16, 32, 64, 128), strides=(1, 2, 2, 2),
                                       leaky=use_leaky, norm=norm_type).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'], strict=False)  # strict=False

    # metrics: Dice score and Hausdorff distance
    dice_metric = DiceMetric(include_background=True, reduction='mean')
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)

    if optimize_overlap:
        post_trans = Compose([Activations(sigmoid=True),
                              AsDiscrete(threshold_values=True),
                              RemoveMinorConnectedComponents(10),
                              ETThresholdSuppression(thr=suppress_thr, is_overlap=True, global_replace=is_global)
                              ])
    else:
        post_trans = Compose([Activations(softmax=True),
                              AsDiscrete(argmax=True),
                              AsDiscrete(to_onehot=True, n_classes=4),
                              RemoveMinorConnectedComponents(10),
                              ETThresholdSuppression(thr=suppress_thr, is_overlap=False, global_replace=is_global)
                              ])

    class_names = ['avg', 'et', 'tc', 'wt']

    # prepare train & validation dataset
    if use_validation_set:
        data_dir = args.root
        subject_reader = InferenceReader(data_dir)
        valset = subject_reader.get_testset()
    elif use_trainset:
        data_dir = './data/MICCAI_BraTS2019_TrainingData'
        subject_reader = InferenceReader(data_dir)
        valset = subject_reader.get_trainset()
    else:
        data_dir = './data/MICCAI_BraTS2019_TrainingData'
        subject_reader = InferenceReader(data_dir)
        valset = subject_reader.get_dataset(test_size=0.2, random_state=dataseed)

    metric_meter = MetricMeter(metrics=['dice', 'hd'], class_names=class_names)
    with torch.no_grad():
        for i, subject in enumerate(valset):
            subject_name = subject['name']
            if patch_test:
                grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap)
                patch_loader = DataLoader(grid_sampler, batch_size=2, num_workers=24)
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')
                for batch_data in patch_loader:
                    inputs_t1 = batch_data['t1'][tio.DATA].to(device)
                    inputs_t2 = batch_data['t2'][tio.DATA].to(device)
                    inputs_t1ce = batch_data['t1ce'][tio.DATA].to(device)
                    inputs_flair = batch_data['flair'][tio.DATA].to(device)
                    locations = batch_data[tio.LOCATION]
                    val_inputs = torch.cat([inputs_t1, inputs_t2, inputs_t1ce, inputs_flair], dim=1)
                    if model_name in ['unet', 'attention_unet', 'panet']:
                        # use TTA
                        if args.tta:
                            val_outputs = model(val_inputs)['out']  # no flip
                            val_outputs += model(val_inputs.flip(dims=(2,)))['out'].flip(dims=(2,))  # flip H
                            val_outputs += model(val_inputs.flip(dims=(3,)))['out'].flip(dims=(3,))  # flip H
                            val_outputs += model(val_inputs.flip(dims=(4,)))['out'].flip(dims=(4,))  # flip W
                            val_outputs += model(val_inputs.flip(dims=(2, 3)))['out'].flip(dims=(2, 3))  # flip D
                            val_outputs += model(val_inputs.flip(dims=(2, 4)))['out'].flip(dims=(2, 4))  # flip H, W
                            val_outputs += model(val_inputs.flip(dims=(3, 4)))['out'].flip(dims=(3, 4))  # flip H, D
                            val_outputs += model(val_inputs.flip(dims=(2, 3, 4)))['out'].flip(dims=(2, 3, 4))  # flip H, W, D
                            val_outputs = val_outputs / 8.0  # mean
                        else:
                            val_outputs = model(val_inputs)['out']
                    else:
                        if args.tta:
                            val_outputs = model(val_inputs)['out'][-1]  # no flip
                            val_outputs += model(val_inputs.flip(dims=(2,)))['out'][-1].flip(dims=(2,))  # flip H
                            val_outputs += model(val_inputs.flip(dims=(3,)))['out'][-1].flip(dims=(3,))  # flip H
                            val_outputs += model(val_inputs.flip(dims=(4,)))['out'][-1].flip(dims=(4,))  # flip W
                            val_outputs += model(val_inputs.flip(dims=(2, 3)))['out'][-1].flip(dims=(2, 3))  # flip D
                            val_outputs += model(val_inputs.flip(dims=(2, 4)))['out'][-1].flip(dims=(2, 4))  # flip H, W
                            val_outputs += model(val_inputs.flip(dims=(3, 4)))['out'][-1].flip(dims=(3, 4))  # flip H, D
                            val_outputs += model(val_inputs.flip(dims=(2, 3, 4)))['out'][-1].flip(dims=(2, 3, 4))  # flip H, W, D
                            val_outputs = val_outputs / 8.0  # mean
                        else:
                            val_outputs = model(val_inputs)['out'][-1]
                    aggregator.add_batch(val_outputs, locations)
                output_tensor = aggregator.get_output_tensor().to(device)
                output_tensor = post_trans(output_tensor.unsqueeze(0))
                if not optimize_overlap:
                    output_tensor = overlap_labels(output_tensor)
                inverse_transform = subject.get_inverse_transform()
                # note that predictions and ground truth are overlapped
                predictions = inverse_transform(output_tensor.squeeze(0)).unsqueeze(0)
            else:
                inputs_t1 = subject['t1'][tio.DATA].unsqueeze(0).to(device)
                inputs_t2 = subject['t2'][tio.DATA].unsqueeze(0).to(device)
                inputs_t1ce = subject['t1ce'][tio.DATA].unsqueeze(0).to(device)
                inputs_flair = subject['flair'][tio.DATA].unsqueeze(0).to(device)
                val_inputs = torch.cat([inputs_t1, inputs_t2, inputs_t1ce, inputs_flair], dim=1)
                val_outputs = post_trans(model(val_inputs)['out'])
                if not optimize_overlap:
                    val_outputs = overlap_labels(val_outputs)

                inverse_transform = subject.get_inverse_transform()
                # note that predictions and ground truth are overlapped
                predictions = inverse_transform(val_outputs.squeeze(0)).unsqueeze(0)

            if not use_validation_set:
                label = subject['label'][tio.DATA].to(device).unsqueeze(0)
                ground_truth = inverse_transform(overlap_labels(label).squeeze(0)).unsqueeze(0)
                # compute dice and hausdorff distance 95
                dice_et, dice_et_not_nan = dice_metric(y_pred=predictions[:, 1:2, ...], y=ground_truth[:, 1:2, ...])
                dice_tc, dice_tc_not_nan = dice_metric(y_pred=predictions[:, 2:3, ...], y=ground_truth[:, 2:3, ...])
                dice_wt, dice_wt_not_nan = dice_metric(y_pred=predictions[:, 3:4, ...], y=ground_truth[:, 3:4, ...])

                hd95_et, hd95_et_not_nan = hausdorff_metric(y_pred=predictions[:, 1:2, ...],
                                                            y=ground_truth[:, 1:2, ...])
                hd95_tc, hd95_tc_not_nan = hausdorff_metric(y_pred=predictions[:, 2:3, ...],
                                                            y=ground_truth[:, 2:3, ...])
                hd95_wt, hd95_wt_not_nan = hausdorff_metric(y_pred=predictions[:, 3:4, ...],
                                                            y=ground_truth[:, 3:4, ...])

                # post-process Dice and HD95 values
                # if subject has no enhancing tumor, empty prediction yields Dice of 1 and HD95 of 0
                # otherwise, false positive yields Dice of 0 and HD95 of 373.13 (worst single case)
                if dice_et_not_nan.item() == 0:
                    if predictions[:, 1:2, ...].max() == 0 and ground_truth[:, 1:2, ...].max() == 0:
                        dice_et = torch.as_tensor(1)
                    else:
                        dice_et = torch.as_tensor(0)

                if hd95_et_not_nan.item() == 0:
                    if predictions[:, 1:2, ...].max() == 0 and ground_truth[:, 1:2, ...].max() == 0:
                        hd95_et = torch.as_tensor(0)
                    else:
                        hd95_et = torch.as_tensor(373.13)

                if hd95_et.item() == np.inf:
                    hd95_et = torch.as_tensor(373.13)

                et_metric = {'et_dice': dice_et.item(), 'et_hd': hd95_et.item()}
                tc_metric = {'tc_dice': dice_tc.item(), 'tc_hd': hd95_tc.item()}
                wt_metric = {'wt_dice': dice_wt.item(), 'wt_hd': hd95_wt.item()}

                avg_dice = (dice_et.item() + dice_tc.item() + dice_wt.item()) / 3
                avg_hd = (hd95_et.item() + hd95_tc.item() + hd95_wt.item()) / 3
                avg_metric = {'avg_dice': avg_dice, 'avg_hd': avg_hd}

                metric = {**et_metric, **tc_metric, **wt_metric, **avg_metric, 'name': subject['name']}
                metric_meter.update(metric)

            # overlapped predictions shall be discretized before saving to numpy
            pred_numpy = discretize_labels(predictions.squeeze(0)).cpu().numpy()
            # nifti_data = nib.Nifti1Image(pred_numpy.astype(np.int), affine=affine_array)
            nifti_data = nib.Nifti1Image(pred_numpy.astype(np.float), affine=affine_array)
            nifti_data.header.get_xyzt_units()
            nifti_data.to_filename(os.path.join(pred_savedir, '{}.nii.gz'.format(subject_name)))  # Save as NiBabel file
            print('{}/{} Inference:{}, done'.format(i + 1, len(valset), subject_name))

            # generate slice 
            if args.snapshot:
                snap_path = os.path.join(pred_savedir, 'snap')
                os.makedirs(snap_path, exist_ok=True)
                H, W, T = 240, 240, 160
                output = pred_numpy
                """ --- grey figure---"""
                # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                """ --- colorful figure--- """
                Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                Snapshot_img[:, :, 2, :][np.where(output == 4)] = 255


                for frame in range(T):
                    slice_save_path = os.path.join(snap_path, subject_name)
                    if not os.path.exists(slice_save_path):
                        os.makedirs(slice_save_path)
                    # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                    imageio.imwrite(os.path.join(slice_save_path, str(frame) + '.png'), Snapshot_img[:, :, :, frame])
                print('{}/{} Snapshot:{}, done'.format(i + 1, len(valset), subject_name))

        metric_meter.save(filename='inference_{}.csv'.format(model_name))


if __name__ == '__main__':
    main()
