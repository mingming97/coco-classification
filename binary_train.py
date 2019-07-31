import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from binary_coco_cls_dataset import CocoClsDataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

DIR_NAME = 'checkpoints/binary_cls'

def parse_args():
    parser = argparse.ArgumentParser(description='COCO cls')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes of data used to fine-tune the pre-trained model')
    parser.add_argument('--print_freq', type=int, default=500)
    # Optimization options
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of the source data.')
    parser.add_argument('--lr', type=float, default=0.01, help='The Learning Rate.')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/binary_cls/checkpoint.pth')
    parser.add_argument('--decay_epoch', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    args = parser.parse_args()
    return args


def load_model(args):
    model = torchvision.models.resnet50(pretrained=True)
    n_features = model.fc.in_features
    model.fc = torch.nn.Linear(n_features, args.num_classes)
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
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr_schedule(optimizer, args.lr, epoch + 1, args.decay_epoch, args.gamma)
        for i, data in enumerate(train_dataloader):
            imgs, target = data
            imgs, target = imgs.cuda(), target.cuda()
            output = model(imgs)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % args.print_freq == 0:
                log('epoch: {} | iter: {} | loss: {:.6f}'.format(epoch + 1, i, loss.item()))

        acc = validate(model, val_dataloader, epoch + 1)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(DIR_NAME, 'checkpoint.pth'))
        log('best_acc: {}'.format(best_acc))


def validate(model, val_dataloader, epoch):
    model.eval()
    correct_num = 0
    no_bg_correct_num = 0
    for imgs, target in val_dataloader:
        imgs, target = imgs.cuda(), target.cuda()
        with torch.no_grad():
            output = model(imgs)
        _, pred = output.max(dim=1)
        _, target = target.max(dim=1)
        inds = torch.nonzero(pred == target).squeeze()
        correct_num += inds.size(0)
        correct_labels = target[inds]
        no_bg_correct_num += correct_labels.sum().item()
    no_bg_num = len(val_dataloader.dataset) - val_dataloader.dataset.num_dict['background']
    log('without_bg_acc: {}({}/{})'.format(no_bg_correct_num/no_bg_num, no_bg_correct_num, no_bg_num))
    acc = correct_num / len(val_dataloader.dataset)
    log('epoch: {} | accuracy: {}/{}, {}'.format(epoch, correct_num, len(val_dataloader.dataset), acc))
    log('-'*15)
    model.train()
    return acc


def log(logstr):
    file_name = os.path.join(DIR_NAME, 'test.log')
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
                                   phase='train')
    val_dataset = CocoClsDataset(root_dir='/home1/share/coco/', 
                                 ann_file='annotations/instances_val2017.json',
                                 img_dir='images/val2017',
                                 bg_bboxes_file='./bg_bboxes/coco_val_bg_bboxes.log',
                                 phase='test')
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=2)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=2)
    model = load_model(args)
    optimizer = load_optimizer(model, args)
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    # train(args, model, optimizer, train_dataloader, val_dataloader)
    validate(model, val_dataloader, 0)
