import tqdm
import time
from tensorboardX import SummaryWriter
from utils import *
from evaluation import *

def val(args, lr, model, dataloader_val, epoch, loss_train_mean, writer):
    print('----------------------val----------------------')
    start = time.time()
    with torch.no_grad():
        model.eval()

        #定义装指标的空列表
        PA_all = []
        recall_all = []
        precision_all = []
        f1_all = []
        miou_all = []
        OA_all = []
        kappa_all = []


        for i, (img1, img2, label) in enumerate(dataloader_val):
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()

            predict = model(img1, img2)

            predict = predict.squeeze()  # squeeze()的作用是去除一个维度
            predict = reverse_one_hot(predict)

            label = label.squeeze()
            label = reverse_one_hot(label)

            pa = Pixel_Accuracy(predict, label)
            recall = Recall(predict, label)
            precision = Precision(predict, label)
            f1 = 2 * recall * precision / (recall + precision + 1e-6)
            miou = mean_IU(predict, label)
            OA = Overall_Accuracy(predict, label)
            kappa = Kappa(predict, label)

            PA_all.append(pa)
            recall_all.append(recall)
            precision_all.append(precision)
            f1_all.append(f1)
            miou_all.append(miou)
            OA_all.append(OA)
            kappa_all.append(kappa)


    #对所有指标做平均
    pa = np.mean(PA_all)
    recall = np.mean(recall_all)
    precision = np.mean(precision_all)
    f1 = np.mean(f1_all)
    OA = np.mean(OA_all)
    kappa = np.mean(kappa_all)
    miou = np.mean(miou_all)


    str_ = ("%15.5g;" * 10) % (epoch+1, lr,  loss_train_mean, pa, recall, precision, f1, OA, kappa,  miou)
    with open(f'{args.results}/{args.data_name}/{args.model_name}.txt', 'a') as f:
        f.write(str_ + '\n')

    #输出参数
    print('PA:          {:}'.format(pa))
    print('Recall:      {:}'.format(recall))
    print('Precision:   {:}'.format(precision))
    print('F1:          {:}'.format(f1))
    print('Miou:        {:}'.format(miou))
    print('OA:          {:}'.format(OA))
    print('kappa:       {:}'.format(kappa))
    print('Time:        {:}s'.format(time.time() - start))

    #可视化
    writer.add_scalar('{}_Pa'.format('val'), pa, epoch+1)
    writer.add_scalar('{}_Recall'.format('val'), recall, epoch + 1)
    writer.add_scalar('{}_Precision'.format('val'), precision, epoch + 1)
    writer.add_scalar('{}_F1'.format('val'), f1, epoch + 1)
    writer.add_scalar('{}_OA'.format('val'), OA, epoch + 1)
    writer.add_scalar('{}_kappa'.format('val'), kappa, epoch + 1)
    writer.add_scalar('{}_Miou'.format('val'), miou, epoch + 1)

    return miou

def train(args, model, optimizer, dataloader_train, dataloader_val, exp_lr_scheduler):
    s = ("%15s;" * 10) % ("epoch", "lr", "loss", "PA", "Recall", "Precision", "F1", "OA", "kappa", "Miou")
    with open(f'{args.results}/{args.data_name}/{args.model_name}.txt', 'a') as file:
        file.write(args.model_name + '\n')
        file.write(s + '\n')
    print('----------------------train----------------------')

    #继承之前最好的miou
    miou_max = args.miou_max
    save_log_path = f'{args.save_log_path}/{args.data_name}/{args.model_name}'
    writer = SummaryWriter(logdir=save_log_path)

    for epoch in range(args.num_epochs):
        model.train()
        exp_lr_scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch+1, lr))
        loss_record = []

        for i, (img1, img2, label) in enumerate(dataloader_train):
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
            if args.warmup == 1 and epoch == 0:
                lr = args.lr / (len(dataloader_train) - i)
                tq.set_description('epoch %d lr %f' % (epoch + 1, lr))

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            output = model(img1, img2)
            loss = torch.nn.BCEWithLogitsLoss()(output, label)

            # output, out1 = model(img1, img2)
            # loss1 = torch.nn.BCEWithLogitsLoss()(output, label)
            # loss2 = torch.nn.BCEWithLogitsLoss()(out1, label)
            # loss = loss1 * 0.6 + loss2 * 0.4

            # output, out1, out2 = model(img1, img2)
            # loss1 = torch.nn.BCEWithLogitsLoss()(output, label)
            # loss2 = torch.nn.BCEWithLogitsLoss()(out1, label)
            # loss3 = torch.nn.BCEWithLogitsLoss()(out2, label)
            # loss = loss1 * 0.4 + loss2 * 0.3 + loss3 * 0.3

            # output, out1, out2, out3 = model(img1, img2)
            # loss1 = torch.nn.BCEWithLogitsLoss()(output, label)
            # loss2 = torch.nn.BCEWithLogitsLoss()(out1, label)
            # loss3 = torch.nn.BCEWithLogitsLoss()(out2, label)
            # loss4 = torch.nn.BCEWithLogitsLoss()(out3, label)
            # # loss = (loss1+loss2+loss3+loss4)/4
            # loss = loss1 * 0.4 + loss2 * 0.2 + loss3 * 0.2 + loss4 * 0.2

            # output, out1, out2, out3, out4 = model(img1, img2)
            # loss1 = torch.nn.BCEWithLogitsLoss()(output, label)
            # loss2 = torch.nn.BCEWithLogitsLoss()(out1, label)
            # loss3 = torch.nn.BCEWithLogitsLoss()(out2, label)
            # loss4 = torch.nn.BCEWithLogitsLoss()(out3, label)
            # loss5 = torch.nn.BCEWithLogitsLoss()(out4, label)
            # loss = loss1*0.4 + loss2*0.15 + loss3*0.15 + loss4*0.15 + loss5*0.15

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        print('Loss for train :{:6f}'.format(loss_train_mean))

        writer.add_scalar('{}_loss'.format('train'), loss_train_mean, epoch + 1)
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            save_path = f'{args.save_model_path}/{args.data_name}/{args.model_name}/' + 'epoch_{:}.pth'.format(epoch)
            torch.save(model.state_dict(), save_path)

        if epoch % args.validation_step == 0:
            miou = val(args, lr, model, dataloader_val, epoch, loss_train_mean, writer)

            if miou > miou_max:
                save_path = f'{args.save_model_path}/{args.data_name}/{args.model_name}/' + 'miou_{:.6f}.pth'.format(miou)
                torch.save(model.state_dict(), save_path)
                miou_max = miou

    writer.close()
    save_path = f'{args.save_model_path}/{args.data_name}/{args.model_name}/' + 'last.pth'
    torch.save(model.state_dict(), save_path)