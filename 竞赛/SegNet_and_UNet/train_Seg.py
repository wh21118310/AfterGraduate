from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from numpy import *
from data_loader.dataset import colorize_mask, fast_hist
from models.seg_net import Segnet
from PIL import Image
import numpy as np

train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=opt.num_workers)

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

net = Segnet(opt.input_nc, opt.output_nc)

if opt.net != '':
    net.load_state_dict(torch.load(opt.netG))
else:
    net.apply(weights_init)
if opt.cuda:
    net.cuda()
if opt.num_GPU > 1:
    net=nn.DataParallel(net)


###########   LOSS & OPTIMIZER   ##########
# criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
###########   GLOBAL VARIABLES   ###########
initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
semantic_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
initial_image = Variable(initial_image)
semantic_image = Variable(semantic_image)

if opt.cuda:
    initial_image = initial_image.cuda()
    semantic_image = semantic_image.cuda()
    criterion = criterion.cuda()

if __name__ == '__main__':
    
    log = open('./checkpoint/Segnet/train_Segnet_log.txt', 'w')
    start = time.time()
    net.train()
    hist = np.zeros((opt.output_nc, opt.output_nc))
    for epoch in range(1, opt.niter+1):
        loader = iter(train_loader)
        for i in range(0, train_datatset_.__len__(), opt.batch_size):
            initial_image_, semantic_image_, name = loader.next()
            initial_image.resize_(initial_image_.size()).copy_(initial_image_)
            semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)
            semantic_image_pred = net(initial_image)
            # semantic_image_pred = semantic_image_pred.view(-1)
            # semantic_image = semantic_image.view(-1)

            ### loss ###
            # from IPython import embed;embed()
            assert semantic_image_pred.size()[2:] == semantic_image.size()[1:]
            loss = criterion(semantic_image_pred, semantic_image.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### evaluate ###
            predictions = semantic_image_pred.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            gts = semantic_image.data[:].squeeze_(0).cpu().numpy()
            hist += fast_hist(label_pred=predictions.flatten(), label_true=gts.flatten(),
                              num_classes=opt.output_nc)
            train_acc = np.diag(hist).sum() / hist.sum()

            ########### Logging ##########
            if i % opt.log_step == 0:
                print('[%d/%d][%d/%d] Loss: %.4f TrainAcc: %.4f' %
                      (epoch, opt.niter, i, len(train_loader) * opt.batch_size, loss.item(), train_acc))
                log.write('[%d/%d][%d/%d] Loss: %.4f TrainAcc: %.4f' %
                          (epoch, opt.niter, i, len(train_loader) * opt.batch_size, loss.item(), train_acc))
            if i % opt.test_step == 0:
                gt = semantic_image[0].cpu().numpy().astype(np.uint8)
                gt_color = colorize_mask(gt)
                predictions = semantic_image_pred.data.max(1)[1].squeeze_(1).cpu().numpy()
                prediction = predictions[0]
                predictions_color = colorize_mask(prediction)
                width, height = opt.size_w, opt.size_h
                save_image = Image.new('RGB', (width * 2, height))
                save_image.paste(gt_color, box=(0 * width, 0 * height))
                save_image.paste(predictions_color, box=(1 * width, 0 * height))
                save_image.save(opt.outf + '/epoch_%03d_%03d_gt_pred.png' % (epoch, i))

        if epoch % opt.save_epoch == 0:
            torch.save(net.state_dict(), '%s/model/netG_%s.pth' % (opt.outf, str(epoch)))

    end = time.time()
    torch.save(net.state_dict(), '%s/model/netG_final.pth' % opt.outf)
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    log.close()