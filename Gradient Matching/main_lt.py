import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import *
import sys


def main():
    log_file = "./log/CIFAR10/r-100-all-1ipc.txt"
    log_file = open(log_file, "w")
    sys.stdout = log_file

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10,
                        help='image(s) per class')
    # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')
    parser.add_argument('--num_exp', type=int, default=5,
                        help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20,
                        help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int,
                        default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1,
                        help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01,
                        help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256,
                        help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256,
                        help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise',
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str,
                        default='data', help='dataset path')
    parser.add_argument('--save_path', type=str,
                        default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str,
                        default='ours', help='distance metric')
    parser.add_argument('--dis_threshold', type=float, default=0.1,
                        help='threshold of the precent for a class to distill')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("running on %s" % args.device)
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    args.save_path = os.path.join(
        args.save_path, args.dataset, 'r-100-all-1ipc')

    os.makedirs(args.save_path, exist_ok=True)

    # The list of iterations when we evaluate models and record results.
    eval_it_pool = np.arange(0, args.Iteration+1, 100).tolist(
    ) if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration]
    print('eval_it_pool: ', eval_it_pool)
    # channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    # TODO
    ''' set the channel,..., testloader for CIFAR10-LT like function get_dataset() in utils.py'''
    if args.dataset == 'CIFAR10':
        config_path = './config/Cifar10lt.yaml'
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        config_path = './config/Cifar100lt.yaml'
        num_classes = 100

    dst_train, dst_test, channel, im_size = get_dataset_from_config(
        args, config_path)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    coarse_label_class = [[] for c in range(num_classes)]

    if args.dataset == "CIFAR10":
        test_images = [torch.unsqueeze(dst_test[i]['img'], dim=0)
                       for i in range(len(dst_test))]
        test_labels = [dst_test[i]['label'] for i in range(len(dst_test))]
        test_images = torch.cat(test_images, dim=0)
        test_labels = torch.tensor(
            test_labels, dtype=torch.long)
        testloader = torch.utils.data.DataLoader(TensorDataset(
            test_images, test_labels), batch_size=256, shuffle=True)

        images_all = [torch.unsqueeze(dst_train[i]['img'], dim=0)
                      for i in range(len(dst_train))]
        labels_all = [dst_train[i]['label'] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
    elif args.dataset == "CIFAR100":
        test_images = [torch.unsqueeze(dst_test[i]['img'], dim=0)
                       for i in range(len(dst_test))]
        test_labels = [dst_test[i]['fine_label'] for i in range(len(dst_test))]
        test_images = torch.cat(test_images, dim=0)
        test_labels = torch.tensor(
            test_labels, dtype=torch.long)
        testloader = torch.utils.data.DataLoader(TensorDataset(
            test_images, test_labels), batch_size=256, shuffle=True)
        images_all = [torch.unsqueeze(dst_train[i]['img'], dim=0)
                      for i in range(len(dst_train))]
        labels_all = [dst_train[i]['fine_label']
                      for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
            coarse_label_class[lab].append(dst_train[i]['coarse_label'])
    # TODO

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    # images_all: [images_num, channel, im_size[0], im_size[1]]
    images_all = torch.cat(images_all, dim=0).to(args.device)
    labels_all = torch.tensor(
        labels_all, dtype=torch.long, device=args.device)

    if args.dataset == "CIFAR10":
        for c in range(num_classes):
            print('class c = %d: %d real images' % (c, len(indices_class[c])))
    elif args.dataset == "CIFAR100":
        for c in range(num_classes):
            # print each class's number of images, class number and coarse label number
            print('class c = %d: %d real images, class number = %d, coarse label number = %d' % (
                c, len(indices_class[c]), c, coarse_label_class[c][0]))
            # check whether the coarse label is the same for all images in the same class
            for i in range(len(coarse_label_class[c])):
                if coarse_label_class[c][i] != coarse_label_class[c][0]:
                    print(
                        'coarse label is not the same for all images in the same class')
                    break

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f' % (
            ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    # TODO
    ''' select all the classes to be distillated '''
    distill_classes = []
    each_class_num = [len(c) for c in indices_class]
    for c, cls_num in enumerate(each_class_num):
        if cls_num > 0:
            distill_classes.append(c)
    # TODO

    ''' initialize the synthetic data '''
    """ image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise') """

    image_syn = torch.randn(size=(len(distill_classes)*args.ipc, channel,
                            im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in distill_classes], dtype=torch.long,
                             requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

    print("image_syn shape: ", image_syn.shape)
    print("label_syn shape: ", label_syn.shape)

    if args.init == 'real':
        print('initialize synthetic data from random real images')
        for i, c in enumerate(distill_classes):
            image_syn.data[i*args.ipc:(i+1)*args.ipc] = get_images(c,
                                                                   args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')

    ''' training '''
    optimizer_img = torch.optim.SGD(
        [image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins' % get_time())

    for it in range(args.Iteration+1):

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' %
                      (args.model, model_eval, it))
                if args.dsa:
                    args.epoch_eval_train = 1000
                    args.dc_aug_param = None
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n',
                          args.dsa_param.__dict__)
                else:
                    # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                    args.dc_aug_param = get_daparam(
                        args.dataset, args.model, model_eval, args.ipc)
                    print('DC augmentation parameters: \n',
                          args.dc_aug_param)

                if args.dsa or args.dc_aug_param['strategy'] != 'none':
                    # Training with data augmentation needs more epochs.
                    args.epoch_eval_train = 1000
                else:
                    args.epoch_eval_train = 300

                # test the net trained on real data
                if it == 0:
                    print("*"*50)
                    print("testing on the original dataset:", flush=True)
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(
                            model_eval, channel, num_classes, im_size).to(args.device)
                        image_real_eval, label_real_eval = copy.deepcopy(
                            images_all.detach()), copy.deepcopy(labels_all.detach())
                        evaluate_synset(
                            it_eval, net_eval, image_real_eval, label_real_eval, testloader, args)

                print("*"*50)
                print("testing on the synthetic dataset:", flush=True)

                accs = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                        args.device)  # get a random model
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                        label_syn.detach())  # avoid any unaware modification

                    _, acc_train, acc_test = evaluate_synset(
                        it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                    accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' %
                      (len(accs), model_eval, np.mean(accs), np.std(accs)), flush=True)

                if it == args.Iteration:  # record the final results
                    accs_all_exps[model_eval] += accs

            ''' visualize and save '''
            save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png' % (
                args.method, args.dataset, args.model, args.ipc, 0, it))
            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            # for ch in range(channel):
            #     image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
            image_syn_vis[image_syn_vis < 0] = 0.0
            image_syn_vis[image_syn_vis > 1] = 1.0
            # Trying normalize = True/False may get better visual effects.
            save_image(image_syn_vis, save_name, nrow=args.ipc)

        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size).to(
            args.device)  # get a random model
        net.train()
        net_parameters = list(net.parameters())
        # optimizer_img for synthetic data
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)
        optimizer_net.zero_grad()
        loss_avg = 0
        # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.
        args.dc_aug_param = None

        for ol in range(args.outer_loop):

            ''' freeze the running mu and sigma for BatchNorm layers '''
            # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
            # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
            # This would make the training with BatchNorm layers easier.

            BN_flag = False
            BNSizePC = 16  # for batch normalization
            for module in net.modules():
                if 'BatchNorm' in module._get_name():  # BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat([get_images(c, BNSizePC)
                                      for c in range(num_classes)], dim=0)
                net.train()  # for updating the mu, sigma of BatchNorm
                output_real = net(img_real)  # get running mu, sigma
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        module.eval()  # fix mu and sigma of every BatchNorm layer

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)
            for i, c in enumerate(distill_classes):
                img_real = get_images(c, args.batch_real)
                lab_real = torch.ones(
                    (img_real.shape[0],), device=args.device, dtype=torch.long) * c
                img_syn = image_syn[i*args.ipc:(i+1)*args.ipc].reshape(
                    (args.ipc, channel, im_size[0], im_size[1]))
                lab_syn = torch.ones(
                    (args.ipc,), device=args.device, dtype=torch.long) * c

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(
                        img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(
                        img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(
                    loss_syn, net_parameters, create_graph=True)

                loss += match_loss(gw_syn, gw_real, args)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()

            if ol == args.outer_loop - 1:
                break

            ''' update network '''
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach(
            )), copy.deepcopy(label_syn.detach())  # avoid any unaware modification

            ''' concat the images_remain and labels_remain to image_syn_eval and label_syn_eval '''
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(
                dst_syn_train, batch_size=args.batch_train, shuffle=True)
            for il in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net,
                      criterion, args, aug=True if args.dsa else False)

        loss_avg /= (num_classes*args.outer_loop)

        if it % 10 == 0:
            print('%s iter = %04d, loss = %.4f' %
                  (get_time(), it, loss_avg), flush=True)

        if it == args.Iteration:  # only record the final results
            data_save.append([copy.deepcopy(
                image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(
                args.save_path, 'res_%s_%s_%s_%dipc.pt' % (args.method, args.dataset, args.model, args.ipc)))

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (
            args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == '__main__':
    main()
