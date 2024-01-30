import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'dropout': (0.0, 'dropout probability'),
        'gpu': (False, 'which cuda device to use (False for cpu training)'),
        'info_section': (40, 'embedding dim = info_section * 3 = 120'),
        'lr': (0.01, 'learning rate'),
        'select_method': ('all_node', '[all_node, end_node]'),
        'num_batch_per_epoch': (5, 'the circle number of each epoch'),
        'epochs': (100, 'number of epoch'),
        'num_thread': (12, 'number of thread'),
        'metapath_length': (3, 'metapath length'),
        'n_limit': (5, 'lambda, neighbour nodes limit'),
        'separate_training': (False, 'separate training two model or not'),
        'h3_method': ('mean', '[mean, sum, cat] cat means concat'),
        'k': (2, 'training times'),
        'k_i': (0, 'current training time'),
        'select_meta': (True, 'automatic select meta path weight')
    },
    'model_config': {
        'task': ('nc', 'task type node classification]'),
        'model': ('HMLP', 'which encoder to use [HMLP]'),
        'dim': (256, 'embedding dimension'),
        'manifold': ('Hyperboloid', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)')
    },
    'data_config': {
        'dataset': ('Kawarith', 'which dataset to use: [Twitter2012, Crisislext, Kawarith]'),
        'train_percent': (20, 'training percentage'),
        'n_classes': (0, 'number of classes')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
