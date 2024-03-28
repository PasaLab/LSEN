import argparse
import os

import dgl
import numpy as np
import torch


def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    norm = torch.pow(in_deg, -0.5)
    norm[torch.isinf(norm)] = 0
    # in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    # norm = 1.0 / in_deg
    return norm

def get_big_graph(backgrounds, num_ents, num_rels):
    if(len(backgrounds) == 0):
        src, dst = np.arange(num_ents), np.arange(num_ents)
        rel = np.zeros(num_ents)
    else:
    # if len(backgrounds) == 0:
    #     return dgl.DGLGraph()
        data = backgrounds
        src, rel, dst = data.transpose()

        loop_nodes = np.arange(num_ents)
        src, dst = np.concatenate((src, dst, loop_nodes)), np.concatenate((dst, src, loop_nodes))
        vec_zeros = np.zeros(num_ents)
        rel = np.concatenate((rel, rel+num_rels, vec_zeros))
    
    g = dgl.DGLGraph()
    g.add_nodes(num_ents)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_ents, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.edata['rel'] = torch.LongTensor(rel)
    # g.edata['type_o'] = torch.LongTensor(rel_o)
    return g

def make_batch(data, s_frequency, o_frequency, times, batch_size):
    for i in range(len(times)):
        l = times[i][0]
        if i < len(times)-1:
            r = times[i+1][0]
        else:
            r = len(data)
        yield [data[l:r], s_frequency[l:r], o_frequency[l:r]]
    
def execute_valid(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                 data, s_frequency, o_frequency, dev_t):
     
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()
    
    valid_loss = 0
    batch_num = 0
    
    for batch_data in make_batch(data, s_frequency, o_frequency, dev_t, args.batch_size):
        triples = batch_data[0][:,:3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()

        with torch.no_grad():
            cur_loss = model(batch_data, his_g, 'Valid', total_data)
            valid_loss += cur_loss.item()
            batch_num += 1
        
        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return valid_loss / batch_num

def execute_test(args, backgrounds, his_g, num_nodes, num_rels, total_data, model,
                 data, s_frequency, o_frequency, test_t):
    
    device = args.device
    total_data = torch.from_numpy(total_data).to(device)
    model.eval()
    
    s_ranks, o_ranks, all_ranks = [], [], []

    for batch_data in make_batch(data, s_frequency, o_frequency, test_t, args.batch_size):
        triples = batch_data[0][:,:3]
        batch_data[0] = torch.from_numpy(batch_data[0]).to(device)
        batch_data[1] = torch.from_numpy(batch_data[1]).to(device).float()
        batch_data[2] = torch.from_numpy(batch_data[2]).to(device).float()
       
        with torch.no_grad():
            sub_rank, obj_rank = model(batch_data, his_g, 'Test', total_data)

            s_ranks += sub_rank
            o_ranks += obj_rank
            tmp = sub_rank + obj_rank
            all_ranks += tmp
            
        g = get_big_graph(triples, num_nodes, num_rels)
        if len(backgrounds) >= args.history_len:
            backgrounds = backgrounds[1:]
            his_g = his_g[1:]
        backgrounds.append(triples)
        g = g.to(device)
        his_g.append(g)

    return s_ranks, o_ranks, all_ranks

def write2file(s_ranks, o_ranks, all_ranks, file_test):
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    print("Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    print("Subject test MR (lk): {:.6f}".format(s_mr_lk))
    file_test.write("Subject test MRR (lk): {:.6f}".format(s_mrr_lk) + '\n')
    file_test.write("Subject test MR (lk): {:.6f}".format(s_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        print("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))
        file_test.write("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk) + '\n')

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    print("Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    print("Object test MR (lk): {:.6f}".format(o_mr_lk))
    file_test.write("Object test MRR (lk): {:.6f}".format(o_mrr_lk) + '\n')
    file_test.write("Object test MR (lk): {:.6f}".format(o_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        print("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))
        file_test.write("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk) + '\n')

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    print("ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    print("ALL test MR (lk): {:.6f}".format(all_mr_lk))
    file_test.write("ALL test MRR (lk): {:.6f}".format(all_mrr_lk) + '\n')
    file_test.write("ALL test MR (lk): {:.6f}".format(all_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        print("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))
        file_test.write("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk) + '\n')
