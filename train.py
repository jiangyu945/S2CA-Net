import os
import yaml
import tqdm
import argparse

import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.networks.utils import one_hot


from models.networks import UNet, UNet_L4, UNet_L4_LKDW, UNet_L4_Deform, \
    UNet_L4_PoolFormer, UNet_L4_ConvFormer, UNet_L4_ConvAttFormer, \
    UNet_L4_ASPMLP, UNet_L4_GateCASPMLP, UNet_L4_Gate_CASPMLP, UNet_L4_Gate_CASPMLP_B, UNet_L4_Former_CSMLP, \
    UNet_L4_Former_CSHMLP, UNet_L4_ConvFormer_CSMLP, UNet_L4_Former_CSMLP_LKDW, UNet_L4_Former_CSMLP_LKDW_Deform,\
    UNet_LGSM_MCA_Deform, UNet_LGSM_MCA_MSADC, UNet_LGSM_MCA_MSADC_Plus, \
    Baseline, UNet_MCA, UNet_MSADC, \
    AttentionUNet, EnhancedUNet, CascadedUNet, PriorAttentionNet
from utils.data_pipeline import SubjectReader, overlap_labels, ETThresholdSuppression, RemoveMinorConnectedComponents
from utils.iterator import MetricMeter, set_random_seed, CosineAnnealingWithWarmUp

import setproctitle
from utils.log import Log

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet_l4',
                        help='network for training, support UNet and FuseUNet')
    parser.add_argument('--dataset', default="BraTS2019", help='dataset name')  # BraTS2019 | BraTS2020

    ## Chose run device
    On_Device = '3090'  # A4000 | 3090 | 3090_Dell
    if On_Device == '3090':
        parser.add_argument('--root', default="/home/ff/dataset/jy/BraTS2019/MICCAI_BraTS_2019_Data_Training_Merge",
                            help='train dataset path')
        # parser.add_argument('--model_save_path', default='/datasets/jy/model_pth', type=str, help='resume training')
        parser.add_argument('--ncpu', type=int, default=4, help='number of workers for dataloader')
    elif On_Device == '3090_Dell':
        # parser.add_argument('--root', default="/home/ff/jy/datasets/BraTS2019/MICCAI_BraTS_2019_Data_Training_Merge",  # 2019
        #                     help='train dataset path')
        parser.add_argument('--root', default="/home/ff/jy/datasets/BraTS2019/MICCAI_BraTS2019_TrainingData",
                            help='train dataset path')
        # parser.add_argument('--model_save_path', default='models', type=str, help='resume training')
        parser.add_argument('--ncpu', type=int, default=4, help='number of workers for dataloader')
    elif On_Device == 'A4000':
        parser.add_argument('--root', default="/usr/jy/code/UNet3d_BraTS/train", help='train dataset path')  # 2019
        # parser.add_argument('--root', default="/home/ff/jy/datasets/Preprocess_BraTS/2019/3D/train", help='train dataset path')
        # parser.add_argument('--model_save_path', default='models', type=str, help='resume training')
        parser.add_argument('--ncpu', type=int, default=4, help='number of workers for dataloader')
    else:
        raise Exception('[error]Device can not fond!')

    parser.add_argument('--interval', type=int, default=10,
                        help='interval for validation')
    parser.add_argument('--mixed', action="store_true",
                        help='whether to use mixed precision training to save video memory')
    parser.add_argument('-c', '--checkpoint', type=str, help='model checkpoint used for breakpoint continuation')
    parser.add_argument('-pc', '--pretrain_ckpt', type=str, help='model checkpoint for training')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='model optimizer, support SGD/Adam')
    parser.add_argument('--benchmark', action="store_true",
                        help='whether to use cudnn benchmark to speed up convolution operations')
    parser.add_argument('--patch_test', action="store_true",
                        help='whether to use patched-pipeline when inference')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs')
    parser.add_argument('--gpu', type=str, default="0", help='GPU No')
    parser.add_argument('--trainset', action='store_true',
                        help='training the model on the entire BraTS Training Set')
    parser.add_argument('--verbose', action='store_true',
                        help='print progress bar when training')
    parser.add_argument('--ds', action='store_true',
                        help='deep supervision')
    parser.add_argument('-cfg', '--config', type=str, default='adam.cfg',
                        help='config file for train parameters')
    return parser.parse_args()

def main():
    args = parse_args()
    model_dict = {'unet': UNet,
                  'unet_l4': UNet_L4,
                  'unet_l4_lkdw': UNet_L4_LKDW,
                  'unet_l4_deform': UNet_L4_Deform,
                  'unet_l4_aspmlp': UNet_L4_ASPMLP,
                  'unet_l4_gate_caspmlp': UNet_L4_Gate_CASPMLP,
                  'unet_l4_gate_caspmlp_b': UNet_L4_Gate_CASPMLP_B,
                  'unet_l4_poolformer': UNet_L4_PoolFormer,
                  'unet_l4_convformer': UNet_L4_ConvFormer,
                  'unet_l4_convattformer': UNet_L4_ConvAttFormer,
                  'unet_l4_former_csmlp': UNet_L4_Former_CSMLP,
                  'unet_l4_former_csmlp_lkdw': UNet_L4_Former_CSMLP_LKDW,
                  'unet_l4_former_csmlp_lkdw_deform': UNet_L4_Former_CSMLP_LKDW_Deform,
                  'unet_l4_convformer_csmlp': UNet_L4_ConvFormer_CSMLP,
                  'unet_l4_former_cshmlp': UNet_L4_Former_CSHMLP,
                  'unet_lgsm_mca_deform': UNet_LGSM_MCA_Deform,
                  'unet_lgsm_mca_msadc': UNet_LGSM_MCA_MSADC,
                  'baseline': Baseline,
                  'unet_mca': UNet_MCA,
                  'unet_msadc': UNet_MSADC,
                  's2ca_net': UNet_LGSM_MCA_MSADC_Plus,
                  'panet': PriorAttentionNet,
                  'attention_unet': AttentionUNet,
                  'enhanced': EnhancedUNet,
                  'cascade': CascadedUNet}

    assert args.model in model_dict.keys(), 'Model name is wrong!'
    assert args.optimizer in ('sgd', 'adam'), 'Optimizer not supported!'

    model_name = args.model
    dataset_name = args.dataset
    eval_interval = args.interval
    is_mixed = args.mixed
    benchmark = args.benchmark
    num_gpu = args.ngpu
    no_gpu = args.gpu
    num_workers = args.ncpu
    use_trainset = args.trainset
    verbose = args.verbose
    patch_test = args.patch_test
    is_ds = args.ds
    # create the save path of model checkpoints
    if use_trainset:
        save_folder = f'ds_{is_ds}'
    else:
        save_folder = f'ds_{is_ds}_val'
    # save_dir = './checkpoints/{}/{}'.format(model_name, save_folder)
    save_dir = f'./checkpoints/{dataset_name}/{model_name}/{save_folder}'
    os.makedirs(save_dir, exist_ok=True)

    print('-' * 30)
    print(f'{dataset_name} Challenge Training!')
    print(f'Model name: {model_name}')
    print('Mixed Precision - {}; CUDNN Benchmark - {}; Num GPU - {}; Num Worker - {}; Patch Test - {}'.format(
        is_mixed, benchmark, num_gpu, num_workers, patch_test))

    # load the cfg file
    if args.optimizer == 'adam':
        cfg_file = f'configs/{args.config}'  # adam.cfg
    else:
        cfg_file = 'configs/sgd.cfg'
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        print("successfully loaded config file: ", cfg)

    seed = cfg['TRAIN']['SEED']  # random seed
    dataseed = cfg['TRAIN']['DATASEED']  # random seed for data split, used for Monte-Carlo cross-validation
    batch_size = cfg['TRAIN']['BATCHSIZE']
    num_epochs = cfg['TRAIN']['EPOCHS']
    lr = cfg['TRAIN']['LR']
    decay = cfg['TRAIN']['DECAY']  # l2 regularization
    warm_up_epochs = cfg['TRAIN']['BURN_IN']  # warm up epochs at the beginning of the training                   
    max_lr_epochs = cfg['TRAIN']['BURN']  # num of epochs with full lr
    momentum = cfg['TRAIN']['MOMENTUM']
    patch_size = cfg['TRAIN']['PATCHSIZE']  # size of each patch, works for patched training
    optimize_overlap = cfg['TRAIN']['OVERLAP']  # whether to optimize directly on overlap regions (et, tc and wt)
    suppress_thr = cfg['TEST']['THR']  # suppress threshold for enhancing tumor suppression
    patch_overlap = cfg['TEST']['PATCHOVERLAP']  # overlap of patches
    is_global = cfg['TEST']['GLOBAL']  # global replacement of et
    use_leaky = cfg['MODEL']['LEAKY']
    norm_type = cfg['MODEL']['NORM']  # type of norm layer (batch/instance/group)
    use_deconv = cfg['MODEL']['DECONV']  # use deconv or interpolation for upsampling
    start_epoch = 0
    # prepare train & validation dataset
    data_dir = args.root

    # set random seed for reproductivity
    set_random_seed(seed=seed, benchmark=benchmark)

    subject_reader = SubjectReader(data_dir, training_size=patch_size)
    if not use_trainset:
        trainset, valset = subject_reader.get_dataset(test_size=0.2, random_state=dataseed)
        val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)
    else:
        trainset = subject_reader.get_trainset()
    train_loader = DataLoader(trainset, batch_size=batch_size * num_gpu, shuffle=True,
                              num_workers=num_workers, multiprocessing_context='spawn')
    # train_loader = DataLoader(trainset, batch_size=batch_size * num_gpu, shuffle=True, num_workers=num_workers)
    device = torch.device(f'cuda:{no_gpu}' if torch.cuda.is_available() else 'cpu')
    # define network
    if model_name in ['unet', 'attention_unet', 'panet']:
        # model = model_dict[model_name](num_classes=4, input_channels=4, use_deconv=use_deconv,
        #                                channels=(32, 64, 128, 256, 320), strides=(1, 2, 2, 2, 2),
        #                                leaky=use_leaky, norm=norm_type).to(device)
        model = model_dict[model_name](num_classes=4, input_channels=4, use_deconv=use_deconv,
                                       leaky=use_leaky, norm=norm_type).to(device)
    else:  # for l4 model
        model = model_dict[model_name](num_classes=4, input_channels=4, use_deconv=use_deconv,
                                       channels=(16, 32, 64, 128), strides=(1, 2, 2, 2),  # (24, 32, 60, 128)
                                       leaky=use_leaky, norm=norm_type).to(device)

    # print('Model name: {}'.format(model_name))
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total trainable parameters: %.2f M.' % (n_params / 1000000.0))

    # from thop import profile
    # x_test = torch.randn([1, 4, 128, 128, 128])
    # flops, params = profile(model, (x_test.to(device),))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, nesterov=True, weight_decay=decay)

    scheduler = CosineAnnealingWithWarmUp(optimizer,
                                          cycle_steps=num_epochs * len(train_loader),
                                          max_lr_steps=max_lr_epochs * len(train_loader),
                                          max_lr=lr,
                                          min_lr=lr/1000,
                                          warmup_steps=warm_up_epochs * len(train_loader))

    if args.pretrain_ckpt:
        pretrain_ckpt = torch.load(args.pretrain_ckpt)
        print('Load pretrained ckeckpoint: {}'.format(args.pretrain_ckpt))
        if model_name == 'panet':
            model.load_state_dict(pretrain_ckpt['net'], strict=False)
        else:
            model.first_stage.load_state_dict(pretrain_ckpt['net'], strict=False)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        print('Load ckpt for breakpoint continuation: {}'.format(args.checkpoint))
        model.load_state_dict(checkpoint['net'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    if num_gpu > 1:
        model = torch.nn.DataParallel(model)
        print('Multi GPU Training. Data Parallel is activated.')

    binary_criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    if optimize_overlap:
        # optimize on overlap labels (et, tc, wt), use sigmoid activation (multi-label classification)
        criterion = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    else:
        # optimize on vanilla labels (et, net/ncr, ed)
        criterion = DiceCELoss(to_onehot_y=False, softmax=True, squared_pred=True)

    # scaler for mixed precision training
    scaler = GradScaler()

    # log init
    log = Log(save_dir, 'log.txt')
    log.init()
    log.write('Config -----')
    for arg in vars(args):
        log.write('%s: %s' % (arg, getattr(args, arg)))
    log.write('------------')

    # metrics: Dice score and Hausdorff distance
    # note that y_pred and y_true must be one-hot encoded and first dim is batch
    dice_metric = DiceMetric(include_background=True, reduction='none')
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction='none', percentile=95)

    # post-processing of predictions
    # including activation and enhancing-tumor suppression
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

    # classes for evaluation, including et, tc, wt and the average
    class_names = ['avg', 'et', 'tc', 'wt']

    # starts training
    for epoch in range(start_epoch + 1, num_epochs + 1):
        # set process name
        setproctitle.setproctitle('{}: {}/{}'.format(model_name, epoch, num_epochs))
        print("Epoch {}/{}".format(epoch, num_epochs))
        model.train()
        epoch_binary_loss = 0
        epoch_multi_loss = 0
        if verbose:
            loader = tqdm.tqdm(train_loader)
        else:
            loader = train_loader
        for step, batch_data in enumerate(loader):
            # load data separately and then concatenate in channel dimension
            # modalities must be augmented separately for better performance
            inputs_t1 = batch_data['t1'].to(device)
            inputs_t2 = batch_data['t2'].to(device)
            inputs_t1ce = batch_data['t1ce'].to(device)
            inputs_flair = batch_data['flair'].to(device)
            inputs = torch.cat([inputs_t1, inputs_t2, inputs_t1ce, inputs_flair], dim=1)  # (B, 4, H, W, D)

            # load targets
            targets = one_hot(batch_data['label'].to(device), num_classes=4)
            binary_targets = torch.sum(targets[:, 1:], dim=1, keepdim=True)

            if optimize_overlap:
                targets = overlap_labels(targets)

            optimizer.zero_grad()
            if is_mixed:
                # automatic mixed precision
                with autocast():
                    outputs = model(inputs)
                    # calculate losses
                    if model_name in ('panet', 'cascade'):
                        binary_loss = binary_criterion(outputs['stage1'], binary_targets)
                    multi_loss = model.get_multi_loss(criterion, outputs, targets, is_ds=is_ds)
                # backward
                if model_name in ('panet', 'cascade'):
                    scaler.scale(binary_loss).backward(retain_graph=True)
                scaler.scale(multi_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if model_name in ('panet', 'cascade'):
                    binary_loss = binary_criterion(outputs['stage1'], binary_targets)
                multi_loss = model.get_multi_loss(criterion, outputs, targets, is_ds)
                if model_name in ('panet', 'cascade'):
                    binary_loss.backward(retain_graph=True)
                multi_loss.backward()
                optimizer.step()

            # for linear warmup, learning rate shall be adjusted after iterations rather than epochs
            scheduler.step()
            if model_name in ('panet', 'cascade'):
                epoch_binary_loss += binary_loss.item()
            current_binary_loss = epoch_binary_loss / (step + 1)
            epoch_multi_loss += multi_loss.item()
            current_multi_loss = epoch_multi_loss / (step + 1)
            if verbose:
                loader.set_postfix_str('lr:{:.8f} - Bloss: {:.6f} - Mloss: {:.6f}'.format(
                    optimizer.param_groups[0]['lr'], current_binary_loss, current_multi_loss))
            else:
                if (step + 1) % 20 == 0:
                    print('lr:{:.8f} - Bloss: {:.6f} - Mloss: {:.6f}'.format(
                        optimizer.param_groups[0]['lr'], current_binary_loss, current_multi_loss))

        print("Epoch: {}; average multi loss: {}".format(epoch, current_multi_loss))

        # Hold-Out
        if not use_trainset and epoch % eval_interval == 0:
            torch.cuda.empty_cache()
            print('Evaluating, plz wait...')
            metric_meter = MetricMeter(metrics=['dice', 'hd'], class_names=class_names)
            model.eval()
            with torch.no_grad():
                for step, batch_data in enumerate(val_loader):
                    inputs_t1 = batch_data['t1'].to(device)
                    inputs_t2 = batch_data['t2'].to(device)
                    inputs_t1ce = batch_data['t1ce'].to(device)
                    inputs_flair = batch_data['flair'].to(device)
                    val_inputs = torch.cat([inputs_t1, inputs_t2, inputs_t1ce, inputs_flair], dim=1)

                    # using patch-based inference or not
                    if patch_test:
                        val_outputs = sliding_window_inference(val_inputs, patch_size, batch_size,
                                                               predictor=model.predictor, overlap=patch_overlap)
                    else:
                        val_outputs = model(val_inputs)['out']

                    predictions = post_trans(val_outputs)
                    if not optimize_overlap:
                        predictions = overlap_labels(predictions)
                    ground_truth = overlap_labels(one_hot(batch_data['label'].to(device), num_classes=4))

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
                            print('Subject {}, contain ET {}, predict correctly, Dice=1'.format(batch_data['name'],
                                                                                                ground_truth[:,
                                                                                                1].max()))
                        else:
                            dice_et = torch.as_tensor(0)
                            print('Subject {}, contain ET {}, predict falsely, Dice=0'.format(batch_data['name'],
                                                                                              ground_truth[:, 1].max()))
                    if hd95_et_not_nan.item() == 0:
                        if predictions[:, 1:2, ...].max() == 0 and ground_truth[:, 1:2, ...].max() == 0:
                            hd95_et = torch.as_tensor(0)
                            print('Subject {}, contain ET {}, predict correctly, HD=0'.format(batch_data['name'],
                                                                                              ground_truth[:, 1].max()))
                        else:
                            hd95_et = torch.as_tensor(373.13)
                            print('Subject {}, contain ET {}, predict falsely, HD=373.13'.format(batch_data['name'],
                                                                                                 ground_truth[:,
                                                                                                 1].max()))
                    if hd95_et.item() == np.inf:
                        hd95_et = torch.as_tensor(373.13)
                        print('Subject {}, contain ET {}, predict falsely, HD=373.13'.format(batch_data['name'],
                                                                                             ground_truth[:, 1].max()))

                    et_metric = {'et_dice': dice_et.item(), 'et_hd': hd95_et.item()}
                    tc_metric = {'tc_dice': dice_tc.item(), 'tc_hd': hd95_tc.item()}
                    wt_metric = {'wt_dice': dice_wt.item(), 'wt_hd': hd95_wt.item()}

                    avg_dice = (dice_et.item() + dice_tc.item() + dice_wt.item()) / 3
                    avg_hd = (hd95_et.item() + hd95_tc.item() + hd95_wt.item()) / 3
                    avg_metric = {'avg_dice': avg_dice, 'avg_hd': avg_hd}

                    metric = {**et_metric, **tc_metric, **wt_metric, **avg_metric, 'name': batch_data['name']}
                    metric_meter.update(metric)
            metric_meter.report(print_stats=True)

            checkpoint = {
                "net": model.module.state_dict() if num_gpu > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, os.path.join(save_dir, '{}_Epoch_{}_Dice_{:.3f}_HD_{:.3f}.pkl'.format(model_name, epoch, avg_dice, avg_hd)))

        # All-Train
        if use_trainset and epoch % eval_interval == 0:
            checkpoint = {
                "net": model.module.state_dict() if num_gpu > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, '{}_Epoch_{}_loss_{:.3f}.pkl'.format(model_name, epoch, current_multi_loss)))

    if not use_trainset:
        metric_meter.save(filename='train_{}.csv'.format(model_name))

    # final_checkpoint = {
    #     "net": model.module.state_dict() if num_gpu > 1 else model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     "epoch": epoch
    # }
    # torch.save(final_checkpoint, os.path.join(save_dir, '{}_checkpoint_final.pkl'.format(model_name)))

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
