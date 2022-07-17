import os
import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from stereo_trainer import StereoTrainer

torch.manual_seed(args.seed)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            if 'Flickr1024' in args.data_train:
                t = StereoTrainer(args, loader, _model, _loss, checkpoint)
            else:
                t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                print(args.model, args.save)
                print('using gpus : {}'.format(args.gpus))
                if hasattr(_model.model, 'update_epoch'):
                    _model.model.update_epoch(t.optimizer.get_last_epoch(), args.epochs)
                t.train()
                t.test()
            print(args.model, args.save)
            print('using gpus : {}'.format(args.gpus))

            checkpoint.done()


if __name__ == '__main__':
    args.gpus = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.n_GPUS = 1
    args.resume = -2
    args.test_only = True
    args.binary_mode = 'binary'
    args.dir_data = '/home/zhangxiao/langzhiqiang/sr_data/'
    # args.data_test = 'Set5+Set14+Urban100+BSDS100+Manga109'
    args.data_test = 'Set5+Set14+BSDS100+Urban100'
    args.data_test = args.data_test.split('+')
    args.n_colors = 1
    args.epochs = 300
    args.decay = '200'
    args.lr = 2e-4
    model_name_prefix = 'EDSR'

    # 性能实验
    # model_names = ['51', '52']
    model_names = ['52']
    ncs = [[64,16], [256,32]]
    scales = ['2','3','4']

    # 消融实验
    # model_names = ['65', '66', '67', '61', '63', '64', '55', '56', '57', '58']  # 除去了51、52
    # model_names = ['org']
    # ncs = [[64,16]]
    # scales = ['4']
    
    # # RDN
    # model_name_prefix = 'RDN'
    # # model_names = ['1', '13', '14', '5', 'org']  # 除去了51、52
    # model_names = ['mrb1']  # 除去了51、52
    # model_names = ['1', '13', '14', '5', 'org']
    # ncs = [[64,8]]
    # scales = ['4']
    
    # # RCAN
    # model_name_prefix = 'RCAN'
    # model_names = ['1', '3', '7', '2', 'org']  # 除去了51、52
    # # model_names = ['mrb1']  # 除去了51、52
    # ncs = [[64,5]]
    # scales = ['4']

    # 可视化
    # edsr
    # model_names = ['65', '66', '67', '61', '63', '64', '51', '55', '56', '52', '57', '58']  # 除去了51、52
    # model_names = ['57', '58', '52', 'org']  # 除去了51、52
    # ncs = [[64,16]]
    # rdn
    # model_name_prefix = 'RCAN'
    # model_names = ['1', '3', '7', '2', 'org']
    # ncs = [[64,5]]
    # scales = ['4']
    # # args.data_test = 'Set14'
    # args.data_test = 'test'
    # args.data_test = args.data_test.split('+')
    # args.save_results = True
    # args.save_gt = True
    # args.need_rgb = True

    for model_name in model_names:
        for nc in ncs:
            for scale in scales:
                args.model = '{}_{}'.format(model_name_prefix, model_name if model_name.find('org')>-1 or model_name.find('mrb')>-1 else 'bireal'+model_name)
                if model_name.find('bicubic')>0:
                    args.model = 'bicubic'
                args.n_resblocks = nc[1]
                args.n_feats = nc[0]
                args.scale = scale
                args.scale = list(map(lambda x: int(x), args.scale.split('+')))
                args.load = '{}_{}_x{}'.format(args.model.lower(), args.binary_mode, int(scale))
                if 'edsr' in args.model.lower():
                    args.load += '_n{}_c{}'.format(args.n_resblocks, args.n_feats)
                elif  'rdn' in args.model.lower():
                    args.load += '_nr{}_nc{}_c{}'.format(args.n_resblocks, args.n_convs, args.n_feats)
                elif  'rcan' in args.model.lower():
                    args.load += '_g{}_n{}_c{}'.format(args.n_resgroups, args.n_resblocks, args.n_feats)
                else:
                    assert False, 'there is no this model : {}'.format(args.model)
                args.load += '_e{}_lr{}_b{}_p{}'.format(args.epochs, args.lr, args.batch_size, args.patch_size)

                print(args.load)
                checkpoint = utility.checkpoint(args)
                main()
                # print()
