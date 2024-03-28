import argparse
import datetime
import os
import pickle
import time

import numpy as np
import torch
from model import NET
from utils import *

"""
seed = 987
np.random.seed(seed)
torch.manual_seed(seed)
"""

def split_by_time(data):
    times = []
    latest_t = -1
    for i, tuple in enumerate(data):
        t = tuple[3]
        if t != latest_t:
            times.append([i, t])
            latest_t = t
    return times

def make_backgrounds(data, data_t, k):
    backgrounds = []
    times = data_t[-k:]
    for i in range(len(times)):
        l = times[i][0]
        if i < len(times)-1:
            r = times[i+1][0]
        else:
            r = len(data)
        backgrounds.append(np.array(data[l:r][:,:3]))
    return backgrounds
    
    
def train(args):
    settings = {}

    num_nodes, num_rels, num_t = get_total_number('./data/' + args.dataset, 'stat.txt')
    train_data, _ = load_quadruples('./data/' + args.dataset, 'train.txt')
    try:
        dev_data, _ = load_quadruples('./data/' + args.dataset, 'valid.txt')
    except:
        print(args.dataset, 'does not have valid set.')
    test_data, _ = load_quadruples('./data/' + args.dataset, 'test.txt')
    try:
        total_data, _ = load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt', 'test.txt')
    except:
        total_data, _ = load_quadruples('./data/' + args.dataset, 'train.txt', 'test.txt')
        print(args.dataset, 'does not have valid set.')

    train_t = split_by_time(train_data)
    if args.use_valid:
        dev_t = split_by_time(dev_data)
    test_t = split_by_time(test_data)

    train_s_frequency_f = '/train_s_frequency.txt'
    train_o_frequency_f = '/train_o_frequency.txt'

    dev_s_frequency_f = '/dev_s_frequency.txt'
    dev_o_frequency_f = '/dev_o_frequency.txt'

    test_s_frequency_f = '/test_s_frequency.txt'
    test_o_frequency_f = '/test_o_frequency.txt'
    test_s_frequency_offline_f = '/test_s_frequency_offline.txt'
    test_o_frequency_offline_f = '/test_o_frequency_offline.txt'

    with open('./data/' + args.dataset + train_s_frequency_f, 'rb') as f:
        train_s_frequency = pickle.load(f).toarray()
    with open('./data/' + args.dataset + train_o_frequency_f, 'rb') as f:
        train_o_frequency = pickle.load(f).toarray()

    if args.use_valid:
        with open('./data/' + args.dataset + dev_s_frequency_f, 'rb') as f:
            dev_s_frequency = pickle.load(f).toarray()
        with open('./data/' + args.dataset + dev_o_frequency_f, 'rb') as f:
            dev_o_frequency = pickle.load(f).toarray()
        
    if args.mode == "online":
        with open('./data/' + args.dataset + test_s_frequency_f, 'rb') as f:
            test_s_frequency = pickle.load(f).toarray()
        with open('./data/' + args.dataset + test_o_frequency_f, 'rb') as f:
            test_o_frequency = pickle.load(f).toarray()
    elif args.mode == "offline":
        with open('./data/' + args.dataset + test_s_frequency_offline_f, 'rb') as f:
            test_s_frequency = pickle.load(f).toarray()
        with open('./data/' + args.dataset + test_o_frequency_offline_f, 'rb') as f:
            test_o_frequency = pickle.load(f).toarray()
    else:
        print("Invalid mode!")
        exit()
    
    device = args.device

    model = NET(num_nodes, num_rels, num_t, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step)
    model = model.to(device)
    now = datetime.datetime.now()
    dt_string = args.description + now.strftime("%d-%m-%Y,%H-%M-%S") + \
                args.dataset + '-EPOCH' + str(args.max_epochs)
    main_dirName = os.path.join(args.save_dir, dt_string)
    if not os.path.exists(main_dirName):
        os.makedirs(main_dirName)

    model_path = os.path.join(main_dirName, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    settings['main_dirName'] = main_dirName
    file_training = open(os.path.join(main_dirName, "training_record.txt"), "w")
    file_training.write("Training Configuration: \n")
    for key in settings:
        file_training.write(key + ': ' + str(settings[key]) + '\n')
    for arg in vars(args):
        file_training.write(arg + ': ' + str(getattr(args, arg)) + '\n')
    print("Start training...")
    file_training.write("Training Start \n")
    file_training.write("===============================\n")
    
    train_backgrounds, dev_backgrounds, test_backgrounds = [], [], []
    if args.use_valid:
        dev_backgrounds = make_backgrounds(train_data, train_t, args.history_len)
        test_backgrounds = make_backgrounds(dev_data, dev_t, args.history_len)
    else:
        test_backgrounds = make_backgrounds(train_data, train_t, args.history_len)
    
    train_his_g, dev_his_g, test_his_g = [], [], []
    if args.use_valid:
        dev_his_g = [get_big_graph(bg, num_nodes, num_rels).to(device) for bg in dev_backgrounds]
    test_his_g = [get_big_graph(bg, num_nodes, num_rels).to(device) for bg in test_backgrounds]

    epoch = 0
    valid_loss_min = float('inf')
    best_epoch = 0
    while epoch < args.max_epochs:
        model.train()
        epoch += 1
        print('$Start Epoch: ', epoch)
        file_training.write('$Start Epoch: ' + str(epoch) + '\n')
        loss_epoch = 0
        time_begin = time.time()
        _batch = 0
        for batch_data in make_batch(train_data, train_s_frequency, train_o_frequency, train_t, args.batch_size):
            triples = np.asarray(batch_data[0][:,:3])
            batch_data[0] = torch.from_numpy(batch_data[0])
            batch_data[1] = torch.from_numpy(batch_data[1]).float()
            batch_data[2] = torch.from_numpy(batch_data[2]).float()
            
            batch_data[0] = batch_data[0].to(device)
            batch_data[1] = batch_data[1].to(device)
            batch_data[2] = batch_data[2].to(device)
            
            batch_loss = model(batch_data, train_his_g, 'Training')
            if batch_loss is not None:
                error = batch_loss
            else:
                continue
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch += error.item()
            _batch += 1
            
            g = get_big_graph(triples, num_nodes, num_rels)
            if len(train_backgrounds) >= args.history_len:
                train_backgrounds = train_backgrounds[1:]
                train_his_g = train_his_g[1:]
            train_backgrounds.append(triples)
            g = g.to(device)
            train_his_g.append(g)
            
        scheduler.step()
        epoch_time = time.time()
        print("Done\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
                format(epoch, loss_epoch / _batch, epoch_time - time_begin))
        file_training.write("******\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
                            format(epoch, loss_epoch / _batch, epoch_time - time_begin) + '\n')
        if args.use_valid and epoch % args.valid_epochs == 0:
            print("Start valid...")
            file_training.write("Start valid...\n")
            valid_loss = execute_valid(args, dev_backgrounds, dev_his_g, num_nodes, num_rels, total_data, model, 
                                       dev_data, dev_s_frequency, dev_o_frequency, dev_t)
            print("Valid loss:", valid_loss)
            file_training.write("Valid loss:" + str(valid_loss) + "\n")
            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                best_epoch = epoch
                torch.save(model, model_path + '/' + args.dataset + '_best.pth')
        if not args.use_valid:
            torch.save(model, model_path + '/' + args.dataset + '_best.pth')
    print("Training done")
    file_training.write("Training done")
    file_training.close()

    # Evaluation
    if args.only_eva:
        dt_string = args.model_dir
        main_dirName = os.path.join(args.save_dir, dt_string)
        model_path = os.path.join(main_dirName, 'models')
        settings['main_dirName'] = main_dirName
    if args.filtering:
        if args.only_eva:
            file_test_path = os.path.join(main_dirName, "test_record_filtering_eva.txt")
        else:
            file_test_path = os.path.join(main_dirName, "test_record_filtering.txt")
    else:
        if args.only_eva:
            file_test_path = os.path.join(main_dirName, "test_record_raw_eva.txt")
        else:
            file_test_path = os.path.join(main_dirName, "test_record_raw.txt")

    file_test = open(file_test_path, "w")
    file_test.write("Testing starts: \n")
    model = torch.load(model_path + '/' + args.dataset + '_best.pth')
    model.eval()
    model.args = args

    if args.dataset != "ICEWS14":
        valid_loss = execute_valid(args, dev_backgrounds, dev_his_g, num_nodes, num_rels, total_data, model,
                                dev_data, dev_s_frequency, dev_o_frequency, dev_t)
        print("***Best Epoch:", best_epoch, "***Minimal Valid loss:", valid_loss)
        file_test.write("***Best Epoch: " + str(best_epoch) + " ***Minimal Valid loss: " + str(valid_loss) + "\n")
    s_ranks1, o_ranks1, all_ranks1 = execute_test(args, test_backgrounds, test_his_g, num_nodes, num_rels, total_data, model, 
                                                  test_data, test_s_frequency, test_o_frequency, test_t)

    # evaluation for link prediction
    write2file(s_ranks1, o_ranks1, all_ranks1, file_test)
    file_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TKGNET')
    parser.add_argument("--description", type=str, default='yago', help="description")
    parser.add_argument("-d", "--dataset", type=str, default='YAGO', help="dataset")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30, help="maximum epochs")
    parser.add_argument("--valid-epochs", type=int, default=1, help="validation epochs")
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha for nceloss")
    parser.add_argument("--lambdax", type=float, default=2.0, help="lambda")
    parser.add_argument("--history-len", type=int, default=1)
    parser.add_argument("--mode", type=str, default="offline")
    parser.add_argument("--graph-layer", type=int, default=2)

    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--lr-dc-step", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.4, help="dropout probability")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--filtering", type=str2bool, default=True)

    parser.add_argument("--only-eva", type=str2bool, default=False, help="whether only evaluation on test set")
    parser.add_argument("--use-valid", type=str2bool, default=True, help="whether using validation set")
    parser.add_argument("--model-dir", type=str, default="", help="model directory")
    parser.add_argument("--save-dir", type=str, default="SAVE", help="save directory")
    parser.add_argument("--eva-dir", type=str, default="SAVE", help="saved dir of the testing model")
    args_main = parser.parse_args()
    print(args_main)
    
    if not os.path.exists(args_main.save_dir):
        os.makedirs(args_main.save_dir)
    train(args_main)
