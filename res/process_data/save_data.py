import torch
import json


def name_file(name, epoch):
    name = {'gen_name': name + f'_gen_{epoch}' + '.pth.tar',
            'disc_name': name + f'_disc_{epoch}' + '.pth.tar',
            'csv_name': name + f'_{epoch}' + '.csv',
            'hist_name': name + '_distance_' + f'{epoch}' + '.png',
            'output_name': name + '_output_' + f'{epoch}' + '.png',
            'gen_json_name': name + f'_gen_{epoch}' + '.json',
            'disc_json_name': name + f'_disc_{epoch}' + '.json'}
    return name


def save_model(gen, disc, name, epoch):
    name_dic = name_file(name, epoch)
    torch.save({'gen_state_dict': gen.state_dict()}, name_dic['gen_name'])
    torch.save({'disc_state_dict': disc.state_dict()}, name_dic['disc_name'])


def save_loss(loss_value_gen, loss_value_disc, path):

    with open(path + '_gen.json', 'w') as fp:
        json.dump(loss_value_gen, fp)
    with open(path + '_disc.json', 'w') as fp:
        json.dump(loss_value_disc, fp)
