import torch
import torch.nn.functional as F


def clip_nll(output_dict, target_dict):
    loss = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])
    return loss


def get_loss_func(loss_type):
    if loss_type == 'clip_nll':
        return clip_nll