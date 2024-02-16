import numpy as np
import torch
from data import HeteData
import random


def enum_longest_metapath_index(name_dict, type_dict, length):
    # Enumerate the longest metapath number list
    hop = []
    for type in type_dict.keys():
        hop.append([type])
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
    return hop


def enum_all_metapath(name_dict, type_dict, length):
    hop = []
    path_list = []
    for type in type_dict.keys():
        hop.append([type])
    path_list.extend(hop)
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
        path_list.extend(hop)
    path_dict = {}
    for path in path_list:
        name = name_dict[path[0]][0]
        for index in path:
            name += name_dict[index][1]
        path_dict[name] = path
    return path_dict


def enum_metapath_name(name_dict, type_dict, length):
    # Enumerate all possible metapath names
    # Results are returned by type
    hop = []
    path_list = []
    result_dict = {}
    for type in type_dict.keys():
        hop.append([type])
        result_dict[name_dict[type][0]] = []
    path_list.extend(hop)
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
        path_list.extend(hop)
    for path in path_list:
        name = name_dict[path[0]][0]
        for index in path:
            name += name_dict[index][1]
        if len(name) > 1:
            result_dict[name[0]].append(name)
    return result_dict


def search_all_path(graph_list, src_node, name_list, metapath_list, metapath_name, path_single_limit=None):
    path_dict = {}
    for path in metapath_list:
        path_dict.update(search_single_path(graph_list, src_node, name_list, path, metapath_name, path_single_limit))
    return path_dict



def search_single_path(graph_list, src_node, name_list, type_sequence, metapath_name, path_single_limit):
    '''
    :param src_nodes: center_node
    :param path_nums: the num of meta_path per node
    :param type_sequence: edge-type sequence without head node
    :return: meta-path list, the n-th element is the n-hop meta-path list.
    '''
    if src_node not in graph_list[type_sequence[0]] or len(graph_list[type_sequence[0]][src_node]) == 0:
        return {}
    path_result = [[[src_node]]]
    hop = len(type_sequence)
    # Perform an adjacency matrix BFS search
    for l in range(hop):
        path_result.append([])
        for list in path_result[l]:
            path_result[l + 1].extend(list_appender(list, graph_list, type_sequence[l], path_single_limit))
    # Limit the search results and save them under Metapath's name
    path_dict = {}
    fullname = metapath_name[type_sequence[0]][0]
    path_dict[fullname[0]] = path_result[0]
    for i in type_sequence:
        fullname += metapath_name[i][1]
    for i in range(len(fullname)):
        if len(path_result[i]) != 0 and fullname[0:i + 1] in name_list[fullname[0]]:
            path_dict[fullname[0:i + 1]] = path_result[i]

    return path_dict


def list_appender(list, graph_list, type, path_limit):
    # Based on each metapath, BFS searches one step further.
    result = []
    if list[-1] not in graph_list[type]: return []

    if path_limit != None and len(graph_list[type][list[-1]]) > path_limit:
        neighbors = random.sample(graph_list[type][list[-1]], path_limit)
    else:
        neighbors = graph_list[type][list[-1]]
    for neighbor in neighbors:
        if neighbor not in list:
            result.append(list + [neighbor])
    return result


def index_to_features(path_dict, x, select_method="all_node"):
    '''
    Convert point sequence numbers into features matrix
    Request space in advance to speed things up
    '''
    result_dict = {}
    for name in path_dict.keys():
        if len(name) == 1:
            result_dict[name] = x[None, path_dict[name][0][0], :]
            result_dict['src_type'] = name
            continue
        np_index = np.array(path_dict[name], dtype=np.int_)
        if select_method == "end_node":
                np_x = np.empty([np_index.shape[0], x.shape[1]])
                np_x[:, 0:x.shape[1]] = x[np_index[:, -1], :]
        else:
            np_x = np.empty([np_index.shape[0], (np_index.shape[1] - 1) * x.shape[1]])
            for i in range(np_index.shape[1] - 1):
                np_x[:, i * x.shape[1]:(i + 1) * x.shape[1]] = x[np_index[:, i + 1], :]
        result_dict[name] = np_x

    return result_dict


def combine_features_dict(list_of_node_dict, batch_src_index, batch_src_label, DEVICE):
    '''
    Stack feature dictionaries of multiple points according to metapath
  
    '''
    metapath_dict = {}
    feature_dict = {}
    row_dict = {}
    column_dict = {}
    type_dict = {}
    tensor_dict = {}
    index_dict = {}
    label_dict = {}
    # First count the number of point types, 
    # and classify the point numbers into categories 
    #and store them in the dictionary.
    for index in range(len(list_of_node_dict)):
        type = list_of_node_dict[index]['src_type']
        if type not in type_dict:
            type_dict[type] = []
            index_dict[type] = []
            label_dict[type] = []

        type_dict[type].append(index)
        label_dict[type].append(batch_src_label[index])
        index_dict[type].append(batch_src_index[index])

    for type in type_dict:
        # Initialize the dictionary of features, tensors and row number records for each class
        metapath_dict[type] = set()
        feature_dict[type] = {}
        tensor_dict[type] = {}
        row_dict[type] = {}
        column_dict[type] = {}
        # Convert the label of each category to Tensor
        label_dict[type] = torch.Tensor(label_dict[type]).long().to(DEVICE)
        for node_index in type_dict[type]:
            # Take the union of the metapaths of each type of points
            metapath_dict[type].update(list_of_node_dict[node_index].keys())
        # Remove redundant 'src_type' key. This key must exist by design.
        metapath_dict[type].remove('src_type')

    for type in type_dict:
        for metapath in metapath_dict[type]:
            # Initialize
            row_dict[type][metapath] = []
            for node_index in type_dict[type]:
                # Count the number of feature rows for each metapath of each point and record the number of feature rows
                if metapath not in list_of_node_dict[node_index]:
                    # This point does not have this metapath, record 0
                    row_dict[type][metapath].append(0)
                else:
                    row_dict[type][metapath].append(list_of_node_dict[node_index][metapath].shape[0])
                    column_dict[type][metapath] = list_of_node_dict[node_index][metapath].shape[1]
            # Initialize the total feature matrix
            # Sum the number of rows
            stack_list = []
            for i in range(len(type_dict[type])):
                if row_dict[type][metapath][i] == 0:
                    # This point does not have the metapath, skip
                    continue
                else:
                    stack_list.append(torch.from_numpy(list_of_node_dict[type_dict[type][i]][metapath]))
            # Finally, use torch.cat to save time
            feature_dict[type][metapath] = torch.cat(stack_list, dim=0).float().to(DEVICE)
    return feature_dict, index_dict, label_dict, row_dict

