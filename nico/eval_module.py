import torch
import torch.nn as nn
import os
import json
import tabulate
import random
import time
from utils import Acc_Per_Context, Acc_Per_Context_Class, cal_acc, save_model, load_model


@torch.no_grad()
def eval_training(config, args, net, test_loader, loss_function, writer, epoch=0, tb=True):
    start = time.time()
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    acc_per_context = Acc_Per_Context(config['cxt_dic_path'])

    for batch_data in test_loader:
        if len(batch_data) == 3:
            images, labels, context = batch_data[0], batch_data[1], batch_data[2]
            processed_images = None
            images = images.cuda()
            labels = labels.cuda()
        elif len(batch_data) == 4:
            images, labels, context, processed_images = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            images = images.cuda()
            labels = labels.cuda()
            processed_images = processed_images.cuda()
        else:
            raise ValueError

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            if 'prime' in args.net:
                outputs = net(images, processed_images)
            else:
                outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        acc_per_context.update(preds, labels, context)

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    print('Evaluate Acc Per Context...')
    acc_cxt = acc_per_context.cal_acc()
    print(tabulate.tabulate(acc_cxt, headers=['Context', 'Acc'], tablefmt='grid'))

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)


@torch.no_grad()
def eval_best(config, args, net, test_loader, loss_function ,checkpoint_path, best_epoch):
    start = time.time()
    try:
        load_model(net, checkpoint_path.format(net=args.net, epoch=best_epoch, type='best'))
        # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=best_epoch, type='best')))
    except:
        print('no best checkpoint')
        load_model(net, checkpoint_path.format(net=args.net, epoch=180, type='regular'))
        # net.load_state_dict(torch.load(checkpoint_path.format(net=args.net, epoch=180, type='regular')))
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    label2train = test_loader.dataset.label2train
    label2train = {v: k for k, v in label2train.items()}
    acc_per_context = Acc_Per_Context_Class(config['cxt_dic_path'], list(label2train.keys()))

    for batch_data in test_loader:
        if len(batch_data) == 3:
            images, labels, context = batch_data[0], batch_data[1], batch_data[2]
            processed_images = None
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
        elif len(batch_data) == 4:
            images, labels, context, processed_images = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
                processed_images = processed_images.cuda()
        else:
            raise ValueError

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            if 'prime' in args.net:
                outputs = net(images, processed_images)
            else:
                outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        acc_per_context.update(preds, labels, context)

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    evaluate_acc_per_context_per_class(config, acc_per_context, label2train)

    return correct.float() / len(test_loader.dataset)


def evaluate_acc_per_context_per_class(config, acc_per_context, label2train):
    print('Evaluate Acc Per Context Per Class...')
    class_dic = json.load(open(config['class_dic_path'], 'r'))
    class_dic = {v: k for k, v in class_dic.items()}
    acc_cxt_all_class = acc_per_context.cal_acc()
    for label_class in acc_cxt_all_class.keys():
        acc_class = acc_cxt_all_class[label_class]
        print('Class: %s' % (class_dic[int(label2train[label_class])]))
        print(tabulate.tabulate(acc_class, headers=['Context', 'Acc'], tablefmt='grid'))


def evaluate_zero_shot_generalization(config, acc_per_context, label2train):
    class_dic = json.load(open(config['class_dic_path'], 'r'))
    class_dic = {v: k for k, v in class_dic.items()}
    acc_cxt_all_class = acc_per_context.cal_acc()

    num_zero_shot_correct = 0
    num_zero_shot_samples = 0
    for label_class in acc_cxt_all_class.keys():
        acc_class = acc_cxt_all_class[label_class]
        for (ctx, acc) in acc_class:
            zero_shot_ctx_list = []
            if ctx not in config['variance_opt']['training_dist'][class_dic[int(label2train[label_class])]]:
                zero_shot_ctx_list.append(ctx)
            correct_info = acc_per_context.count_correct_info_in_some_contexts_of_a_label(label_class, zero_shot_ctx_list)
            num_zero_shot_correct += correct_info['num_correct']
            num_zero_shot_samples += correct_info['num_all_samples']

    zero_shot_acc = num_zero_shot_correct / num_zero_shot_samples
    print('The zero shot generalization accuracy is {}'.format(zero_shot_acc))


@torch.no_grad()
def eval_mode(config, args, net, test_loader, loss_function, model_path=None, contain_zeroshot=True):
    start = time.time()
    if model_path is not None:
        load_model(net, model_path)
    if isinstance(net, list):
        for net_ in net:
            net_.eval()
    else:
        net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    label2train = test_loader.dataset.label2train
    label2train = {v: k for k, v in label2train.items()}
    acc_per_context = Acc_Per_Context_Class(config['cxt_dic_path'], list(label2train.keys()))

    for batch_data in test_loader:
        if len(batch_data) == 3:
            images, labels, context = batch_data[0], batch_data[1], batch_data[2]
            key_inputs = None
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
        elif len(batch_data) == 4:
            images, labels, context, key_inputs = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
                key_inputs = key_inputs.cuda()
        else:
            raise ValueError

        if isinstance(net, list):
            feature = net[-1](images)
            if isinstance(feature, list):
                feature = feature[0]  # chosse causal feature
            try:
                W_mean = torch.stack([net_.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            except:
                W_mean = torch.stack([net_.module.fc.weight for net_ in net[:config['variance_opt']['n_env']]], 0).mean(0)
            outputs = nn.functional.linear(feature, W_mean)
        else:
            if 'prime' in args.net:
                outputs = net(images, key_inputs)
            else:
                outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        acc_per_context.update(preds, labels, context)

    finish = time.time()

    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))

    evaluate_acc_per_context_per_class(config, acc_per_context, label2train)
    if contain_zeroshot:
        evaluate_zero_shot_generalization(config, acc_per_context, label2train)

    return correct.float() / len(test_loader.dataset)