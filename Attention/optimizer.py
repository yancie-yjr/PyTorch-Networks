import torch
from torch.optim import lr_scheduler

class Optimizer():
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, net):
        net_parameters_id = {}
        for pname, p in net.named_parameters():
            if pname in ['conv1_1.weight','conv1_2.weight',
                         'conv2_1.weight','conv2_2.weight',
                         'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                         'conv4_1.weight','conv4_2.weight','conv4_3.weight']:

                if 'conv1-4.weight' not in net_parameters_id:
                    net_parameters_id['conv1-4.weight'] = []
                net_parameters_id['conv1-4.weight'].append(p)
            elif pname in ['conv1_1.bias','conv1_2.bias',
                           'conv2_1.bias','conv2_2.bias',
                           'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                           'conv4_1.bias','conv4_2.bias','conv4_3.bias']:

                if 'conv1-4.bias' not in net_parameters_id:
                    net_parameters_id['conv1-4.bias'] = []
                net_parameters_id['conv1-4.bias'].append(p)
            elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:

                if 'conv5.weight' not in net_parameters_id:
                    net_parameters_id['conv5.weight'] = []
                net_parameters_id['conv5.weight'].append(p)
            elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :

                if 'conv5.bias' not in net_parameters_id:
                    net_parameters_id['conv5.bias'] = []
                net_parameters_id['conv5.bias'].append(p)
            elif 'down' in pname:

                if 'conv_down_1-5.weight' not in net_parameters_id:
                    net_parameters_id['conv_down_1-5.weight'] = []

                if 'conv_down_1-5.bias' not in net_parameters_id:
                    net_parameters_id['conv_down_1-5.bias'] = []
                if 'weight' in pname:
                    net_parameters_id['conv_down_1-5.weight'].append(p)

                elif 'bias' in pname:
                    net_parameters_id['conv_down_1-5.bias'].append(p)


            elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                           'score_dsn4.weight','score_dsn5.weight', 'upsample4.weight', 'upsample8.weight', 'upsample16.weight', 'upsample32.weight']:

                if 'score_dsn_1-5.weight' not in net_parameters_id:
                    net_parameters_id['score_dsn_1-5.weight'] = []
                net_parameters_id['score_dsn_1-5.weight'].append(p)
            elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                           'score_dsn4.bias','score_dsn5.bias']:

                if 'score_dsn_1-5.bias' not in net_parameters_id:
                    net_parameters_id['score_dsn_1-5.bias'] = []
                net_parameters_id['score_dsn_1-5.bias'].append(p)


        for pname, p in net.attention.named_parameters():
            if 'attn.weight' not in net_parameters_id:
                net_parameters_id['attn.weight'] = []
            if 'attn.bias' not in net_parameters_id:
                net_parameters_id['attn.bias'] = []
            if 'weight' in pname:
                net_parameters_id['attn.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['attn.bias'].append(p)

        for pname, p in net.nl1.named_parameters():
            if 'nl12.weight' not in net_parameters_id:
                net_parameters_id['nl12.weight'] = []
            if 'nl12.bias' not in net_parameters_id:
                net_parameters_id['nl12.bias'] = []
            if 'weight' in pname:
                net_parameters_id['nl12.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['nl12.bias'].append(p)

        for pname, p in net.nl2.named_parameters():
            if 'nl23.weight' not in net_parameters_id:
                net_parameters_id['nl23.weight'] = []
            if 'nl23.bias' not in net_parameters_id:
                net_parameters_id['nl23.bias'] = []
            if 'weight' in pname:
                net_parameters_id['nl23.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['nl23.bias'].append(p)

        for pname, p in net.nl3.named_parameters():
            if 'nl34.weight' not in net_parameters_id:
                net_parameters_id['nl34.weight'] = []
            if 'nl34.bias' not in net_parameters_id:
                net_parameters_id['nl34.bias'] = []
            if 'weight' in pname:
                net_parameters_id['nl34.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['nl34.bias'].append(p)

        for pname, p in net.nl4.named_parameters():
            if 'nl45.weight' not in net_parameters_id:
                net_parameters_id['nl45.weight'] = []
            if 'nl45.bias' not in net_parameters_id:
                net_parameters_id['nl45.bias'] = []
            if 'weight' in pname:
                net_parameters_id['nl45.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['nl45.bias'].append(p)

        for pname, p in net.nl5.named_parameters():
            if 'nl56.weight' not in net_parameters_id:
                net_parameters_id['nl56.weight'] = []
            if 'nl56.bias' not in net_parameters_id:
                net_parameters_id['nl56.bias'] = []
            if 'weight' in pname:
                net_parameters_id['nl56.weight'].append(p)
            elif 'bias' in pname:
                net_parameters_id['nl56.bias'].append(p)

        optim = torch.optim.SGD([
                {'params': net_parameters_id['conv1-4.weight']      , 'lr': self.cfg.lr*0.01    , 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['conv1-4.bias']        , 'lr': self.cfg.lr*0.02    , 'weight_decay': 0.},
                {'params': net_parameters_id['conv5.weight']        , 'lr': self.cfg.lr*1.  , 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['conv5.bias']          , 'lr': self.cfg.lr*2.  , 'weight_decay': 0.},
                {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': self.cfg.lr*0.1  , 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': self.cfg.lr*0.2  , 'weight_decay': 0.},
                {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': self.cfg.lr*0.01 , 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': self.cfg.lr*0.02 , 'weight_decay': 0.},
                {'params': net_parameters_id['attn.weight']  , 'lr': self.cfg.lr*1., 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['attn.bias']    , 'lr': self.cfg.lr*2., 'weight_decay': 0.},
####add by YJR
                {'params': net_parameters_id['nl12.weight']  , 'lr': self.cfg.lr*0.1, 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['nl12.bias']    , 'lr': self.cfg.lr*0.2, 'weight_decay': 0.},
                {'params': net_parameters_id['nl23.weight']  , 'lr': self.cfg.lr*0.1, 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['nl23.bias']    , 'lr': self.cfg.lr*0.2, 'weight_decay': 0.},
                {'params': net_parameters_id['nl34.weight']  , 'lr': self.cfg.lr*0.1, 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['nl34.bias']    , 'lr': self.cfg.lr*0.2, 'weight_decay': 0.},
                {'params': net_parameters_id['nl45.weight']  , 'lr': self.cfg.lr*0.1, 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['nl45.bias']    , 'lr': self.cfg.lr*0.2, 'weight_decay': 0.},
                {'params': net_parameters_id['nl56.weight'], 'lr': self.cfg.lr * 0.1, 'weight_decay': self.cfg.wd},
                {'params': net_parameters_id['nl56.bias'], 'lr': self.cfg.lr * 0.2, 'weight_decay': 0.},
            ], lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd)

        scheduler = lr_scheduler.StepLR(optim, step_size=self.cfg.stepsize, gamma=self.cfg.gamma)

        return optim, scheduler