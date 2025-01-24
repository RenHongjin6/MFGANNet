import os, argparse
import torch
from data_load import Dataset
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from train import train

torch.cuda.current_device()

# 忽视警告
import warnings
warnings.filterwarnings('ignore')


# 对模型中的所有超参数进行定义
def main(model, params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs',            type=int,   default=200,               help='None')
    parser.add_argument('--checkpoint_step',       type=int,   default=5,                 help='save model for every X time')
    parser.add_argument('--validation_step',       type=int,   default=1,                 help='check model for every X time')
    parser.add_argument('--batch_size',            type=int,   default=8,                 help='None')
    parser.add_argument('--num_workers',           type=int,   default=4,                 help='None')
    parser.add_argument('--lr',                    type=float, default=0.0001,            help='None')
    parser.add_argument('--lr_scheduler',          type=int,   default=3,                 help='update the learning rate every X times')
    parser.add_argument('--lr_scheduler_gamma',    type=float, default=0.99,              help='learning rate attenuation coefficient')
    parser.add_argument('--warmup',                type=int,   default=1,                 help='warm up')
    parser.add_argument('--warmup_num',            type=int,   default=1,                 help='warm up the number')
    parser.add_argument('--cuda',                  type=str,   default='0',               help='GPU ids used for training')
    parser.add_argument('--DataParallel',          type=int,   default=1,                 help='train in multi GPU')
    parser.add_argument('--beta1',                 type=float, default=0.5,               help='momentum1 in Adam')
    parser.add_argument('--beta2',                 type=float, default=0.999,             help='momentum2 in Adam')
    parser.add_argument('--miou_max',              type=float, default=0.8,               help='If Miou greater than it, Miou will be saved and update it')
    parser.add_argument('--crop_height',           type=int,   default=256,               help='None')
    parser.add_argument('--crop_width',            type=int,   default=256,               help='None')
    parser.add_argument('--pretrained_model_path', type=str,   default=None,              help='path to save model')
    parser.add_argument('--save_log_path',         type=str,   default='./logs/',         help='path to save the log')
    parser.add_argument('--save_model_path',       type=str,   default='./checkpoints/',  help='path to save model')
    parser.add_argument('--data',                  type=str,   default='./datasets/data', help='path of training data')
    # parser.add_argument('--data',                  type=str,   default='./datasets/LEVIR', help='path of training data')
    parser.add_argument('--data_name',             type=str,   default=None,              help='name of data')
    parser.add_argument('--model_name',            type=str,   default=None,              help='name of model')
    parser.add_argument('--results',               type=str,   default='./results/',      help='path to save result')
    args = parser.parse_args(params)

    # 生成表格去记录超参信息并输出
    tb = PrettyTable(['Num', 'Key', 'Value'])
    args_str = str(args)[10:-1].split(',')
    for i, key_value in enumerate(args_str):
        key, value = key_value.split('=')[0], key_value.split('=')[1]
        tb.add_row([i + 1, key, value])
    print(tb)

    # 查看路径是否存在，不存在则生成路径

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    miou_path = os.listdir(args.save_model_path)
    if args.data_name not in miou_path:
        os.mkdir(f'{args.save_model_path}/{args.data_name}')
    miou_path_list = os.listdir(f'{args.save_model_path}/{args.data_name}')
    if args.model_name not in miou_path_list:
        os.mkdir(f'{args.save_model_path}/{args.data_name}/{args.model_name}')

    if not os.path.exists(args.save_log_path):
        os.makedirs(args.save_log_path)
    log_path = os.listdir(args.save_log_path)
    if args.data_name not in log_path:
        os.mkdir(f'{args.save_log_path}/{args.data_name}')
    log_path_list = os.listdir(f'{args.save_log_path}/{args.data_name}')
    if args.model_name not in log_path_list:
        os.mkdir(f'{args.save_log_path}/{args.data_name}/{args.model_name}')

    if not os.path.exists(args.results):
        os.makedirs(args.results)
    if not os.path.exists(args.results + args.data_name):
        os.makedirs(args.results + args.data_name)

    # 创建训练数据集和验证数据集的路径
    train_path_img1 = os.path.join(args.data, 'train/img1')
    train_path_img2 = os.path.join(args.data, 'train/img2')
    train_path_label = os.path.join(args.data, 'train/label')

    val_path_img1 = os.path.join(args.data, 'test/img1')
    val_path_img2 = os.path.join(args.data, 'test/img2')
    val_path_label = os.path.join(args.data, 'test/label')

    csv_path = os.path.join(args.data, 'class_dict.csv')

    # 读取数据集
    dataset_train = Dataset(
        train_path_img1,
        train_path_img2,
        train_path_label,
        csv_path,
        mode='train'
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    dataset_val = Dataset(
        val_path_img1,
        val_path_img2,
        val_path_label,
        csv_path,
        mode='val'
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 设置模型和参数
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    torch.backends.cudnn.benchmark = True

    model = model()

    if args.DataParallel == 1:
        print('mulit Cuda! cuda:{:}'.format(args.cuda))
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    else:
        print('single Cuda!')
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr, [args.beta1, args.beta2])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler, gamma=args.lr_scheduler_gamma)

    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)  # %S将替换为后边的路径

        # 下载权重
        pretrained_dict = torch.load(args.pretrained_model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Done!')

        # 开始训练
    train(args, model, optimizer, dataloader_train, dataloader_val, exp_lr_scheduler)


if __name__ == '__main__':

    from models.UNet import UNet

    # backbone
    params = [
        '--num_epochs', '200',
        '--batch_size', '4',
        '--lr', '0.0015',
        '--warmup', '0',
        '--lr_scheduler_gamma', '0.95',
        '--lr_scheduler', '4',
        '--miou_max', '0.75',
        '--DataParallel', '1',  # 1: True  0:False
        '--cuda', '0',  # model put in the cuda[0]
        '--checkpoint_step', '10',
        # '--pretrained_model_path', 'checkpoints/data/BiSeNetv1/miou_0.782544.pth',
        '--data_name', 'data',
        '--model_name', 'UNet',
    ]

    main(UNet, params)
