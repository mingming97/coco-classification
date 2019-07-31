import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from multi_head_coco_cls_dataset import CocoClsDataset
import resnet

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

DIR_NAME = 'multi_head_checkpoints/full_sample_1fc'
LOG_NAME = 'full_sample_1fc.log'
FG_WEIGHT = 1.0
BG_WEIGHT = 1.0
TWO_FC = False

def parse_args():
    parser = argparse.ArgumentParser(description='COCO cls')
    parser.add_argument('--num_classes', type=int, default=80,
                        help='number of classes of data used to fine-tune the pre-trained model')
    parser.add_argument('--print_freq', type=int, default=200)
    # Optimization options
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of the source data.')
    parser.add_argument('--lr', type=float, default=0.01, help='The Learning Rate.')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--decay_epoch', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.1)
    args = parser.parse_args()
    return args


def load_model(args):
    model = resnet.resnet50(pretrained=True, two_fc=TWO_FC)
    n_features = model.fc.in_features
    model.fc = torch.nn.Linear(n_features, args.num_classes)
    model.bin_fc = torch.nn.Linear(n_features, 2)
    if TWO_FC:
        model.fc0 = torch.nn.Linear(n_features, n_features)
        model.bin_fc0 = torch.nn.Linear(n_features, n_features)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    return model.cuda()


def load_optimizer(model, args):
    param_group = []
    for k, v in model.named_parameters():
        if not k.__contains__('fc'):
            param_group += [{'params': v, 'lr': args.lr / 10}]
        else:
            param_group += [{'params': v, 'lr': args.lr}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=0.0005)
    return optimizer


def lr_schedule(optimizer, base_lr, epoch, decay_epoch, gamma):
    if epoch == decay_epoch:
        new_lr = base_lr * gamma
        for i in range(len(optimizer.param_groups)):
            if i < len(optimizer.param_groups) - 1:
                optimizer.param_groups[i]['lr'] = new_lr / 10
            else:
                optimizer.param_groups[i]['lr'] = new_lr


def train(args, model, optimizer, train_dataloader, val_dataloader):
    criterion = nn.CrossEntropyLoss()
    bin_criterion = nn.BCEWithLogitsLoss()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr_schedule(optimizer, args.lr, epoch + 1, args.decay_epoch, args.gamma)
        for i, data in enumerate(train_dataloader):
            imgs, bin_target, ce_target = data
            imgs, bin_target, ce_target = imgs.cuda(), bin_target.cuda(), ce_target.cuda()
            bin_output, ce_output = model(imgs)
            bin_loss = bin_criterion(bin_output, bin_target)
            _, bin_label = bin_target.max(dim=1)
            fg_inds = bin_label == 1
            ce_output = ce_output[fg_inds]
            ce_target = ce_target[fg_inds]
            ce_loss = criterion(ce_output, ce_target)
            loss = FG_WEIGHT * ce_loss + BG_WEIGHT * bin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % args.print_freq == 0:
                log('epoch: {} | iter: {} | bin_loss: {:.6f} | ce_loss: {:.6f} | loss: {:.6f}'.format(
                    epoch + 1, i, bin_loss.item(), ce_loss.item(), loss.item()))

        acc = validate(model, val_dataloader, epoch + 1)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(DIR_NAME, 'checkpoint.pth'))
        log('best_acc: {}'.format(best_acc))


def validate(model, val_dataloader, epoch):
    model.eval()
    correct_num = 0
    correct_dict = {}
    binary_correct_num = 0
    for imgs, bin_target, ce_target in val_dataloader:
        imgs, bin_target, ce_target = imgs.cuda(), bin_target.cuda(), ce_target.cuda()
        with torch.no_grad():
            bin_output, ce_output = model(imgs)
        for i in range(len(ce_target)):
            bin_out = bin_output[i].cpu().numpy()
            bin_tar = bin_target[i].cpu().numpy()
            ce_out = ce_output[i].cpu()
            ce_tar = ce_target[i].item() + 1
            if bin_out[0] > bin_out[1]:
                if bin_tar[0] > bin_tar[1]:
                    num = correct_dict.get('background', 0)
                    correct_dict['background'] = num + 1
                    correct_num += 1
                    binary_correct_num += 1
            elif bin_tar[0] < bin_tar[1]:
                binary_correct_num += 1
                _, ind = ce_out.max(dim=0)
                label = ind.item() + 1
                if label == ce_tar:
                    cat_id = val_dataloader.dataset.label2id[label]
                    cat_name = val_dataloader.dataset.id2cat[cat_id]
                    num = correct_dict.get(cat_name, 0)
                    correct_dict[cat_name] = num + 1
                    correct_num += 1

    for cat, num in correct_dict.items():
        log('{}: {}/{}'.format(cat, num, val_dataloader.dataset.num_dict[cat]))
    no_bg_correct_num = correct_num - correct_dict['background']
    no_bg_num = len(val_dataloader.dataset) - val_dataloader.dataset.num_dict['background']
    log('without_bg_acc: {}({}/{})'.format(no_bg_correct_num/no_bg_num, no_bg_correct_num, no_bg_num))
    log('binary_bg_acc: {}({}/{})'.format(binary_correct_num/len(val_dataloader.dataset), binary_correct_num, len(val_dataloader.dataset)))
    acc = correct_num / len(val_dataloader.dataset)
    log('epoch: {} | accuracy: {}/{}, {}'.format(epoch, correct_num, len(val_dataloader.dataset), acc))
    log('-'*15)
    model.train()
    return acc


def log(logstr):
    file_name = os.path.join(DIR_NAME, LOG_NAME)
    with open(file_name, 'a') as f:
        print(logstr)
        f.write(logstr)
        f.write('\n')
        f.flush()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    train_dataset = CocoClsDataset(root_dir='/home1/share/coco/', 
                                   ann_file='annotations/instances_train2017.json',
                                   img_dir='images/train2017',
                                   bg_bboxes_file='./bg_bboxes/coco_train_bg_bboxes.log',
                                   phase='train',
                                   less_sample=False,
                                   eq=True)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=2)

    val_dataset = CocoClsDataset(root_dir='/home1/share/coco/', 
                                 ann_file='annotations/instances_val2017.json',
                                 img_dir='images/val2017',
                                 bg_bboxes_file='./bg_bboxes/coco_val_bg_bboxes.log',
                                 phase='test')
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=2)
    model = load_model(args)
    optimizer = load_optimizer(model, args)
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    log('TWO_FC: {}'.format(TWO_FC))
    log('FG_WEIGHT: {}'.format(FG_WEIGHT))
    log('BG_WEIGHT: {}'.format(BG_WEIGHT))
    train(args, model, optimizer, train_dataloader, val_dataloader)

    # validate(model, val_dataloader, 0)
