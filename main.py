import os
import warnings
import argparse
import random
import numpy
import torch
import torch.optim as optim
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter

from model import IDCM_NN
from Loss import CMSP_out
from train_model import train_model
from evaluate import fx_calc_map_label
from data_load import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params', default='parameter/wiki.yaml')
    # parser.add_argument('--dataset', type=str, default='wiki', help='pascal, wiki, nuswide, xmedia')
    # # loss 
    # parser.add_argument('--loss_type', type=str, default='CMSP_out', help='CMSP_out')
    # parser.add_argument('--n_view', type=int, default=2)
    # parser.add_argument('--alpha_p', type=float, default=1)
    # parser.add_argument('--alpha_d', type=float, default=1)  
    # parser.add_argument('--alpha_m', type=float, default=1) 
    # parser.add_argument('--margin', type=float, default=0.1)
    # # opt
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--loss_lr', type=float, default=1e-4)
    # # train
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--epoch', type=int, default=200)
    # parser.add_argument('--dropout', type=float, default=0)
    # parser.add_argument('--eb_size', type=int, default=512)
    # parser.add_argument('--exam_name', type=str, default='wiki')
    # args = parser.parse_args()
    arg = parser.parse_args()

    with open(arg.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    args = EasyDict(params)

    seed = args.seed
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DATA_DIR = 'data/' + args.dataset + '/'
    log_name = args.dataset + '/'  + str(args.loss_type) + "_" + args.exam_name
    writer = SummaryWriter(log_dir=os.path.join('./log', log_name))
    log = open(os.path.join('./log', log_name, 'log.csv'), 'w')
    parameter = 'dataset={}, loss_type={},\n alpha_p={}, alpha_d={}, alpha_m={}, margin={}\n loss_lr={}, batch_size={},  eb_size={}, seed={}, ' \
                'epoch={}, lr={}, dropout={}\n'.format(args.dataset, args.loss_type, str(args.alpha_p), str(args.alpha_d), str(args.alpha_m),
                                         str(args.margin), str(args.loss_lr), str(args.batch_size), str(args.eb_size),
                                           str(args.seed), str(args.epoch), str(args.lr), str(args.dropout))
    log.write(parameter)
    print('...Data loading is beginning...')
    data_loader, input_data_par = load_data(args=args)

    print('...Data loading is completed...')

    model_ft = IDCM_NN(dropout=args.dropout, img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                       pre_dim=input_data_par['num_class'], feat_dim=args.eb_size).to(device)


    if args.loss_type == 'CMSP_out':
        loss = CMSP_out(nb_classes=input_data_par['num_class'], sz_embedding=args.eb_size, mrg=args.margin,
                             alpha=args.alpha_p, beta=args.alpha_d, gamma=args.alpha_m)
        optimizer_loss = optim.SGD(loss.parameters(), lr=args.loss_lr)
        # optimizer_loss = optim.Adam(loss.parameters(), lr=args.loss_lr)
    # elif args.loss_type == 'CMSP_in':
    #     loss = CMSP_in(nb_classes=input_data_par['num_class'], sz_embedding=args.eb_size, mrg=args.margin,
    #                          alpha=args.alpha_p, beta=args.alpha_d, gamma=args.alpha_m)
    #     optimizer_loss = optim.Adam(loss.parameters(), lr=args.loss_lr, weight_decay=0.0001)
    else:
        warnings.warn("loss is no list")

    # Observe that all parameters are being optimized
    params_to_update = list(model_ft.parameters())
    optimizer = optim.Adam(params_to_update, lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0001)
    # optimizer = optim.Adam(params_to_update, lr=args.lr)



    print('...Training is beginning...')
    # Train and evaluate
    model_ft = train_model(model_ft, data_loader, optimizer, log, loss, optimizer_loss, args, writer)
    print('...Training is completed...')
    torch.save(model_ft.state_dict(), os.path.join('./log', log_name, 'model_save.pkl'))

    print('...Evaluation on testing data...')
    view1_feature, view2_feature, view1_predict, view2_predict = model_ft(torch.tensor(input_data_par['img_test']).to(device), torch.tensor(input_data_par['text_test']).to(device))
    label = torch.argmax(torch.tensor(input_data_par['label_test']), dim=1)
    numpy.savetxt(os.path.join('./log', log_name, 'label.tsv'), label)
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    view1_predict = view1_predict.detach().cpu().numpy()
    view2_predict = view2_predict.detach().cpu().numpy()

    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))
    log.write('...Image to Text MAP = {}'.format(img_to_txt) + '\n')

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))
    log.write('...Text to Image MAP = {}'.format(txt_to_img) + '\n')

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
    log.write('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)) + '\n')
    log_sum = open(os.path.join('./log', args.dataset, 'map.csv'), 'a')
    log_sum.write('{},{},{},{}'.format(img_to_txt, txt_to_img, ((img_to_txt + txt_to_img) / 2.), (args.exam_name)) + '\n')

    writer.close()

