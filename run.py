import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import metapath as mp
import sklearn.metrics as sm
import time
from multiprocessing.dummy import Pool as ThreadPool
from models.model import GraphHAM
from models.model import Discriminator
from data import HeteData
from config import parser
from models.hyper_models import NCModel


def node_search_wrapper(index):
    return mp.search_all_path(graph_list, index, metapath_name, metapath_list, data.get_metapath_name(),
                              single_path_limit)


def train(t_model, model, args):


    def index_to_feature_wrapper(dict):
        return mp.index_to_features(dict, data.x, args.select_method)

    # When dividing the train index, the sample points have been selected, and then the metapath is found based on node_search_wrapper.

    start_select = 50
    train_index = train_list[:, 0].tolist()
    train_label = train_list[:, 1]
    train_label = torch.LongTensor(train_label).to(DEVICE)
    print("Loading dataset with thread pool...")
    train_metapath = pool.map(node_search_wrapper, train_index)
    train_features = pool.map(index_to_feature_wrapper, train_metapath)
    val_index = val_list[:, 0].tolist()
    val_label = val_list[:, 1]
    val_metapath = pool.map(node_search_wrapper, val_index)
    val_features = pool.map(index_to_feature_wrapper, val_metapath)
    lr = learning_rate
    model.train()
    t_model.train()
    separate_training = args.separate_training
    h3_method = args.h3_method

    best_h_micro_f1 = 0
    best_h_macro_f1 = 0
    best_t_micro_f1 = 0
    best_t_macro_f1 = 0
    best_c_micro_f1 = 0
    best_c_macro_f1 = 0
    Loss_h=0
    Loss_h_total = []
    c_train_loss_total = []
    c_val_loss_total = []
    t_train_loss_total = []
    t_val_loss_total = []
    h_train_loss_total = []
    h_val_loss_total = []

    if h3_method == 'cat' and not args.separate_training:
        weights = torch.Tensor([1.] * output_dim * 2).to(DEVICE)
    else:
        weights = torch.Tensor([1.] * output_dim).to(DEVICE)



    type_set = set()
    metapath_set = {}
    for node in val_metapath:
        for key in node:
            type_set.add(key[0])
    for type in type_set:
        metapath_set[type] = set()
    for node in val_metapath:
        for key in node:
            if len(node) - 1 > len(metapath_set[key[0]]):
                if len(key) > 1:
                    metapath_set[key[0]].add(key)

    metapath_label = {}
    metapath_onehot = {}
    discriminator = {}
    d_optimizer = {}
    label = {}

    for type in type_set:
        metapath_label[type] = {}
        metapath_onehot[type] = {}
        label[type] = []
        for i, metapath in enumerate(metapath_set[type]):
            metapath_label[type][metapath] = torch.zeros(batch_size, data.type_num, device=DEVICE)
            metapath_onehot[type][metapath] = torch.zeros(batch_size, device=DEVICE).long()
            metapath_onehot[type][metapath][:] = i
            for all_type in data.node_dict:
                if all_type in metapath[1:]:
                    metapath_label[type][metapath][:, data.node_dict[all_type]] = 1
            label[type].append(metapath_label[type][metapath])
        label[type] = torch.cat(label[type], dim=0)
        discriminator[type] = Discriminator(info_section, data.type_num).to(DEVICE)
        d_optimizer[type] = optim.Adam(discriminator[type].parameters(), lr=0.01, weight_decay=5e-4)
    optimizer1 = optim.Adam(t_model.parameters(), lr=lr, weight_decay=5e-4)
    optimizer2 = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    select_flag = False
    time1 = time.time()

    start_time = time.time()

    for e in range(args.epochs):

        # learning text features
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        t_embedding = t_model.encode(data.t_feat.to(DEVICE))
        h = t_model.decode(t_embedding)
        t_output = nn.functional.log_softmax(h[train_index], dim=1)
        if separate_training:
            Loss_t = nn.functional.cross_entropy(t_output, train_label, weights)
            l_t = Loss_t
            t_train_loss_total.append(float(l_t))
            Loss_t.backward()
            optimizer1.step()

        # if single_path_limit is not None and (e + 1) % 20 == 0:
        #     print("Re-sampling...")
        #     train_metapath = pool.map(node_search_wrapper, train_index)
        #     train_features = pool.map(index_to_feature_wrapper, train_metapath)
        h_output = []
        h_pred = []
        h_true = []
        if separate_training:
            for batch in range(num_batch_per_epoch):
                batch_src_choice = np.random.choice(range(train_list.shape[0]), size=(batch_size,), replace=False)
                batch_src_index = train_list[batch_src_choice, 0]
                batch_src_label = train_list[batch_src_choice, 1]
                # batch_feature_list = [train_features[i] for i in batch_src_choice]
                batch_feature_list = []
                for i in batch_src_choice:
                    if train_features[i] != {}:
                        batch_feature_list.append(train_features[i])

                batch_train_feature_dict, batch_src_index_dict, batch_src_label_dict, batch_train_rows_dict = mp.combine_features_dict(
                    batch_feature_list,
                    batch_src_index,
                    batch_src_label, DEVICE)

                for type in d_optimizer:
                    d_optimizer[type].zero_grad()
                if e >= start_select and args.select_meta:
                    batch_train_logits_dict, GAN_input = model(batch_train_rows_dict, batch_train_feature_dict, True,
                                                               DEVICE)
                else:
                    batch_train_logits_dict, GAN_input = model(batch_train_rows_dict, batch_train_feature_dict,
                                                               False, DEVICE)  # 获取模型的输出

                for type in batch_train_logits_dict:
                    Loss_h = criterion(batch_train_logits_dict[type], batch_src_label_dict[type])
                    l_h = Loss_h
                    Loss_h_total.append(float(l_h))
                    Loss_h.backward()
                    d_optimizer[type].step()
                optimizer2.step()

            if len(Loss_h_total) == 0:
                h_train_loss_total.append(-1)
            else:
                h_train_loss_total.append(np.mean(Loss_h_total))
        else:
            train_feature_dict, train_index_dict, train_label_dict, train_rows_dict = mp.combine_features_dict(
                train_features,
                train_index, train_label,
                DEVICE)
            if e >= start_select and args.select_meta:
                train_logits_dict, _ = model(train_rows_dict, train_feature_dict, True)
            else:
                train_logits_dict, _ = model(train_rows_dict, train_feature_dict, False)
            for type in train_logits_dict:
                h_output.append(train_logits_dict[type])
                h_pred.extend(train_logits_dict[type].max(1)[1].cpu().numpy().tolist())  # 预测标签：对预测结果按行取argmax
                h_true.extend(train_label_dict[type].cpu().numpy().tolist())

            h_output = torch.cat(h_output, dim=0)
            # c_output = torch.cat((h_output, t_output), 1)
            if h3_method == 'mean':
                c_output = (h_output + t_output) / 2
            elif h3_method == 'sum':
                c_output = (h_output + t_output)
            else:
                c_output = torch.cat((h_output, t_output), dim=1)

            Loss = nn.functional.cross_entropy(c_output, train_label, weights)
            c_loss = Loss
            c_train_loss_total.append(float(c_loss))
            Loss.backward()
            optimizer1.step()
            optimizer2.step()

        if e >= start_select and select_flag == False and args.select_meta == True:
            select_flag = True
            pretrain_convergence = time2 - time1
            print("Start select! Best f1-score reset to 0.")
            print("Pretrain convergence time:", pretrain_convergence)
            time1 = time.time()
            best_h_micro_f1 = 0
            best_h_macro_f1 = 0
            best_c_micro_f1 = 0
            best_c_macro_f1 = 0

        if select_flag:
            h_micro_f1, h_macro_f1, t_micro_f1, t_macro_f1, c_micro_f1, c_macro_f1, c_v_loss, h_v_loss, t_v_loss \
                = val(t_model, model, val_features, val_index, val_label, weights,True)
            model.show_metapath_importance()
        else:
            h_micro_f1, h_macro_f1, t_micro_f1, t_macro_f1, c_micro_f1, c_macro_f1, c_v_loss, h_v_loss, t_v_loss \
                = val(t_model, model, val_features, val_index, val_label, weights)

        c_val_loss_total.append(c_v_loss)
        t_val_loss_total.append(t_v_loss)
        h_val_loss_total.append(h_v_loss)

        if separate_training:
            if t_micro_f1 >= best_t_micro_f1:
                if t_micro_f1 > best_t_micro_f1:
                    best_t_micro_f1 = t_micro_f1
                    best_t_macro_f1 = t_macro_f1
                    time2 = time.time()
                    if select_flag:
                        torch.save(t_model.state_dict(), "checkpoint/" + dataset + "_best_val_t")
                elif t_macro_f1 > best_t_macro_f1:
                    best_t_micro_f1 = t_micro_f1
                    best_t_macro_f1 = t_macro_f1
                    time2 = time.time()
                    if select_flag:
                        torch.save(t_model.state_dict(), "checkpoint/" + dataset + "_best_val_t")
            if h_micro_f1 >= best_h_micro_f1:
                if h_micro_f1 > best_h_micro_f1:

                    best_h_micro_f1 = h_micro_f1
                    best_h_macro_f1 = h_macro_f1
                    time2 = time.time()
                    if select_flag:
                        torch.save(model.state_dict(), "checkpoint/" + dataset + "_best_val")
                elif h_macro_f1 > best_h_macro_f1:
                    best_h_micro_f1 = h_micro_f1
                    best_h_macro_f1 = h_macro_f1
                    time2 = time.time()
                    if select_flag:
                        torch.save(model.state_dict(), "checkpoint/" + dataset + "_best_val")

        else:
            if c_micro_f1 >= best_c_micro_f1:
                if c_micro_f1 > best_c_micro_f1:
                    best_c_micro_f1 = c_micro_f1
                    best_c_macro_f1 = c_macro
                    time2 = time.time()
                    if select_flag:
                        torch.save(t_model.state_dict(), "checkpoint/" + dataset + "_best_val_t")
                        torch.save(model.state_dict(), "checkpoint/" + dataset + "_best_val")
                elif c_macro > best_c_macro_f1:
                    best_c_micro_f1 = c_micro_f1
                    best_c_macro_f1 = c_macro
                    time2 = time.time()
                    if select_flag:
                        torch.save(t_model.state_dict(), "checkpoint/" + dataset + "_best_val_t")
                        torch.save(model.state_dict(), "checkpoint/" + dataset + "_best_val")

        select_convergence = time2 - time1
        if separate_training:
            print("Epoch ", e, ",Val h_Micro_f1 is ", h_micro_f1, ", h_Macro_F1 is ", h_macro_f1,
                  ", t_Micro_f1 is ", t_micro_f1, ", t_Macro_F1 is ", t_macro_f1,  ', h_Loss:', Loss_h, ', t_Loss:', Loss_t,
                  ", c_Micro_f1 is ", c_micro_f1, ", c_Macro_F1 is", c_macro_f1)

        else:
            print("Epoch ", e, ",Val h_Micro_f1 is ", h_micro_f1, ", h_Macro_F1 is ", h_macro_f1,
                  ", t_Micro_f1 is ", t_micro_f1, ", t_Macro_F1 is ", t_macro_f1, ', Loss:', Loss,
                  ", c_Micro_f1 is ", c_micro_f1, ", c_Macro_F1 is", c_macro_f1)
        print("Select convergence time:", select_convergence)

    end_time = time.time()
    total_time = end_time - start_time
    print('Total time:', total_time)

    torch.save(model.state_dict(), "checkpoint/" + dataset + "_final")

    return total_time, best_c_micro_f1, best_c_macro_f1, \
        best_t_micro_f1, best_t_macro_f1, best_h_micro_f1, best_h_macro_f1

def val(t_model, model, val_features, val_index, val_label, weights, start_select=False):

    h3_method = args.h3_method
    val_feature_dict, val_index_dict, val_label_dict, val_rows_dict = mp.combine_features_dict(val_features,
                                                                                               val_index, val_label,
                                                                                               DEVICE)
    model.eval()
    t_model.eval()
    with torch.no_grad():
        t_embedding = t_model.encode(data.t_feat.to(DEVICE))
        h = t_model.decode(t_embedding)
        t_output = nn.functional.log_softmax(h[val_index], dim=1)
        t_pred = t_output.max(1)[1].cpu().numpy().tolist()

        val_logits_dict, _ = model(val_rows_dict, val_feature_dict, start_select)


        h_pred = []
        h_true = []
        h_output= []
        for type in val_logits_dict:
            h_output.append(val_logits_dict[type])
            h_pred.extend(val_logits_dict[type].max(1)[1].cpu().numpy().tolist())  # 预测标签：对预测结果按行取argmax
            h_true.extend(val_label_dict[type].cpu().numpy().tolist())  # 计算在测试节点/数据上的准确率

        h_output = torch.cat(h_output, dim=0)

        if h3_method == 'mean':
            c_output = (h_output + t_output) / 2
        elif h3_method == 'sum':
            c_output = (h_output + t_output)
        else:
            c_output = torch.cat((h_output, t_output), dim=1)
        c_pred = c_output.max(1)[1].cpu().numpy()

        val_label_tensor = torch.LongTensor(val_label).to(DEVICE)
        c_v_loss = float(nn.functional.cross_entropy(c_output, val_label_tensor, weights))
        h_v_loss = float(nn.functional.cross_entropy(h_output, val_label_tensor, weights))
        t_v_loss = float(nn.functional.cross_entropy(t_output, val_label_tensor, weights))

        h_micro_f1 = sm.f1_score(h_true, h_pred, average='micro')
        h_macro_f1 = sm.f1_score(h_true, h_pred, average='macro')


        t_micro_f1 = sm.f1_score(h_true, t_pred, average='micro')
        t_macro_f1 = sm.f1_score(h_true, t_pred, average='macro')



        c_micro_f1 = sm.f1_score(h_true,c_pred, average='micro')
        c_macro_f1 = sm.f1_score(h_true, c_pred, average='macro')

        return h_micro_f1, h_macro_f1, t_micro_f1, t_macro_f1, c_micro_f1, \
            c_macro_f1, c_v_loss, h_v_loss, t_v_loss


def test(t_model, model, batch_size=200, test_method="best_val"):
    h3_method = args.h3_method
    model.load_state_dict(torch.load("checkpoint/" + dataset + "_" + test_method))
    t_model.load_state_dict(torch.load("checkpoint/" + dataset + "_" + test_method + "_t"))

    def index_to_feature_wrapper(dict):
        return mp.index_to_features(dict, data.x)

    test_index = test_list[:, 0].tolist()
    print("Loading dataset with thread pool...")
    time1 = time.time()
    test_metapath = pool.map(node_search_wrapper, test_index)
    test_features = pool.map(index_to_feature_wrapper, test_metapath)
    time2 = time.time()
    print("Dataset Loaded. Time consumption:", time2 - time1)

    h_pred = []
    h_true = []
    h_output = []

    model.eval()
    t_model.eval()
    test_time_start = time.time()
    with torch.no_grad():
        # text embedding
        t_embedding = t_model.encode(data.t_feat.to(DEVICE))
        h = t_model.decode(t_embedding)
        t_output = nn.functional.log_softmax(h[test_index], dim=1)
        t_pred = t_output.max(1)[1].cpu().numpy()

        batch = 0
        while batch < len(test_index):
            end = batch + batch_size if batch + batch_size <= len(test_index) else len(test_index)
            batch_test_index = test_list[batch:end, 0]
            batch_test_label = test_list[batch:end, 1]
            batch_feature_list = [test_features[i] for i in range(batch, end)]

            batch_test_feature_dict, batch_test_index_dict, batch_test_label_dict, batch_test_rows_dict = mp.combine_features_dict(
                batch_feature_list,
                batch_test_index,
                batch_test_label, DEVICE)
            batch += batch_size
            if args.select_meta:
                batch_test_logits_dict, _ = model(batch_test_rows_dict, batch_test_feature_dict, True)
            else:
                batch_test_logits_dict, _ = model(batch_test_rows_dict, batch_test_feature_dict, False)
            for type in batch_test_logits_dict:
                h_output.append(batch_test_logits_dict[type])
                h_pred.extend(batch_test_logits_dict[type].max(1)[1].cpu().numpy().tolist())
                h_true.extend(batch_test_label_dict[type].cpu().numpy().tolist())
        h_output = torch.cat(h_output, dim=0)

        if h3_method == 'mean':
            c_output = (h_output + t_output) / 2
        elif h3_method == 'sum':
            c_output = (h_output + t_output)
        else:
            c_output = torch.cat((h_output, t_output), dim=1)
        c_pred = c_output.max(1)[1].cpu().numpy()

        t_time = time.time() - test_time_start

        h_micro_f1 = sm.f1_score(h_true, h_pred, average='micro')
        h_macro_f1 = sm.f1_score(h_true, h_pred, average='macro')


        t_micro_f1 = sm.f1_score(h_true, t_pred, average='micro')
        t_macro_f1 = sm.f1_score(h_true, t_pred, average='macro')


        c_micro_f1 = sm.f1_score(h_true, c_pred,average='micro')
        c_macro_f1 = sm.f1_score(h_true, c_pred, average='macro')
        
        print("Final F1 @ " + test_method + ":")
        print("h_Micro_F1:\t", h_micro_f1, "\th_Macro_F1:", h_macro_f1,
              "\tt_Micro_F1:", t_micro_f1, "\t t_Macro_F1:", t_macro_f1,
              "\tc_Micro_F1:", c_micro_f1, "\t c_Macro_F1:", c_macro_f1,)

        model.show_metapath_importance()

        return c_micro_f1, c_macro_f1, t_time


if __name__ == '__main__':


    args = parser.parse_args()
    if torch.cuda.is_available() and args.gpu:
        DEVICE = 'cuda'
        args.device = DEVICE
    elif torch.backends.mps.is_available() and args.gpu:
        DEVICE = 'mps'
        args.device = DEVICE

    else:
        DEVICE = 'cpu'

    print('Training use:{}'.format(DEVICE))

    dataset = args.dataset
    if len(sys.argv) > 1:
        train_percent = int(sys.argv[1])
    else:
        train_percent = args.train_percent
    shuffle = False
    metapath_length = args.metapath_length
    mlp_settings = {'layer_list': [], 'dropout_list': [], 'activation': 'sigmoid'}
    info_section = args.info_section # total embedding dim = info_section *3 = 120
    learning_rate = args.lr
    select_method = args.select_method
    single_path_limit = args.n_limit  # lambda = 5


    num_batch_per_epoch = args.num_batch_per_epoch  # Number of batches per epoch cycle
    batch_size = args.train_percent//20 * 96


    data_processiong_time  = time.time()
    data = HeteData(dataset=dataset, train_percent=train_percent)
    print('Data processing time:{}'.format(time.time() - data_processiong_time))
    graph_list = data.get_dict_of_list()
    input_dim = data.x.shape[1]

    # Number of hidden unit nodes: two layers
    pre_embed_dim = data.type_num * info_section
    output_dim = max(data.train_list[:, 1].tolist()) + 1
    args.n_classes = output_dim
    args.feat_dim = data.t_feat.shape[1]


    assert select_method in ["end_node", "all_node"]

    x = data.x
    train_list = data.train_list
    test_list = data.test_list
    val_list = data.val_list
    criterion = nn.CrossEntropyLoss().to(DEVICE)


    #num_thread = 12
    pool = ThreadPool(args.num_thread)

    # Enumerate metapath and initialize selection model
    metapath_name = mp.enum_metapath_name(data.get_metapath_name(), data.get_metapath_dict(), metapath_length)
    metapath_list = mp.enum_longest_metapath_index(data.get_metapath_name(), data.get_metapath_dict(), metapath_length)




    # result record
    c_micro = []
    c_macro = []
    v_c_micro = []
    v_c_macro = []
    v_t_micro = []
    v_t_macro = []
    v_h_micro = []
    v_h_macro = []
    total_train_time = []
    total_test_time = []
    best_test_embeddings = 0
    best_test_labels = 0
    best_c_test_f1 = 0

    for i in range(args.k):
        args.k_i = i
        select_model = GraphHAM(metapath_list=metapath_name, input_dim=input_dim, pre_embed_dim=pre_embed_dim,
                                select_dim=output_dim, mlp_settings=mlp_settings, mean_test=False,
                                endnode_test=False).to(DEVICE)
        t_model = NCModel(args).to(DEVICE)
        train_time, best_c_micro_f1, best_c_macro_f1, best_t_micro_f1, best_t_macro_f1, best_h_micro_f1, best_h_macro_f1\
            = train(t_model=t_model, model=select_model, args=args)
        c_micro_f1, c_macro_f1, test_time = test(t_model=t_model, model=select_model, test_method="best_val")

        c_micro.append(c_micro_f1)
        c_macro.append(c_macro_f1)
        v_c_micro.append(best_c_micro_f1)
        v_c_macro.append(best_c_macro_f1)
        v_t_micro.append(best_t_micro_f1)
        v_t_macro.append(best_t_macro_f1)
        v_h_micro.append(best_h_micro_f1)
        v_h_macro.append(best_h_macro_f1)
        total_train_time.append(train_time)
        total_test_time.append(test_time)

    c_micro_mean = np.mean(c_micro)
    c_micro_std = np.std(c_micro, ddof=1)
    c_macro_mean = np.mean(c_macro)
    c_macro_std = np.std(c_macro, ddof=1)
    total_train_time_mean = np.mean(total_train_time)
    total_test_time_mean = np.mean(total_test_time)

    v_c_micro_m = np.mean(v_c_micro)
    v_c_micro_std = np.std(v_c_micro, ddof=1)
    v_c_macro_m = np.mean(v_c_macro)
    v_c_macro_std = np.std(v_c_macro, ddof=1)

    v_t_micro_m = np.mean(v_t_micro)
    v_t_micro_std = np.std(v_t_micro, ddof=1)
    v_t_macro_m = np.mean(v_t_macro)
    v_t_macro_std = np.std(v_t_macro, ddof=1)

    v_h_micro_m = np.mean(v_h_micro)
    v_h_micro_std = np.std(v_h_micro, ddof=1)
    v_h_macro_m = np.mean(v_h_macro)
    v_h_macro_std = np.std(v_h_macro, ddof=1)


    print("C_Micro:{} + {}, C_Macro: {} + {}, train_time: {}, test_time: {}".format(c_micro_mean,
                                                                                                c_micro_std,
                                                                                                c_macro_mean,
                                                                                                c_macro_std,
                                                                                                total_train_time_mean,
                                                                                                total_test_time_mean))
    print("val: C_Micro:{} + {}, C_Macro: {} + {},".format(v_c_micro_m, v_c_micro_std,
                                                            v_c_macro_m, v_c_macro_std))
    print("val: t_Micro:{} + {}, t_Macro: {} + {},".format(v_t_micro_m, v_t_micro_std,
                                                           v_t_macro_m, v_t_macro_std))
    print("val: h_Micro:{} + {}, h_Macro: {} + {},".format(v_h_micro_m, v_h_micro_std,
                                                           v_h_macro_m, v_h_macro_std))