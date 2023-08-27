import torch
from nets.unet_training import CE_Loss, Dice_Loss, Focal_Loss, Dice_Score, Dice_loss, gradient_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, dice_loss, focal_loss, cls_weights, num_classes, best_dice, save_path):
    best_dice_one_epoch = best_dice 
    total_loss = 0
    # total_f_score = 0
    total_dice_score = 0
    val_loss = 0
    # val_f_score = 0
    val_dice_score = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch
            weights = torch.from_numpy(cls_weights)

            with torch.no_grad():
                # imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                # print(imgs.shape)
                # pngs = torch.from_numpy(pngs).long()
                # print(pngs.shape)
                # labels = torch.from_numpy(labels).type(torch.FloatTensor)
                # print(labels.shape)
                # weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                """
                交叉熵损失函数使用one-hot编码，不是原始数据，
                但是pytorch的CrossEntropyLoss要求的输入就是每个像素的类别信息
                """
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            ce_focal_loss = loss
            if dice_loss:
                main_dice = Dice_Loss(outputs, labels)
                #main_dice = Dice_loss(outputs, labels)
                # loss = loss + main_dice
                loss = main_dice
            else:
                main_dice = torch.zeros([1])
            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                # _f_score = f_score(outputs, labels)
                dice_score = Dice_Score(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # total_f_score += _f_score.item()
            total_dice_score += dice_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'CE/Focal_loss': ce_focal_loss.item(),
                                'dice_loss': main_dice.item(),
                                'dice_score': total_dice_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    iteration_train = iteration
    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch
            weights = torch.from_numpy(cls_weights)

            with torch.no_grad():
                # imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                # print(imgs.shape)
                # pngs = torch.from_numpy(pngs).long()
                # print(pngs.shape)
                # labels = torch.from_numpy(labels).type(torch.FloatTensor)
                # print(labels.shape)
                # weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                ce_focal_loss = loss
                if dice_loss:
                    main_dice = Dice_Loss(outputs, labels)
                    # main_dice = Dice_loss(outputs, labels)
                    # loss = loss + main_dice
                    loss = main_dice
                else:
                    main_dice = torch.zeros([1])
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                # _f_score = f_score(outputs, labels)
                dice_score = Dice_Score(outputs, labels)

                val_loss += loss.item()
                # val_f_score += _f_score.item()
                val_dice_score += dice_score.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'CE/Focal_loss': ce_focal_loss.item(),
                                'dice_loss': main_dice.item(),
                                'dice_score': val_dice_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1), total_dice_score / (iteration_train + 1), val_dice_score / (iteration + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    # if val_loss / (epoch_step_val + 1) <= 0.13:
    # if True:
    #     torch.save(model.state_dict(), '../autodl-tmp/weights_pspnet/ep%03d-loss%.3f-val_loss%.3f.pth' % (
    #     (epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    dice_coefficient = (val_dice_score / (iteration + 1) + total_dice_score / (iteration_train + 1)) / 2
    if dice_coefficient > best_dice_one_epoch:
        best_dice_one_epoch = dice_coefficient
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dice': dice_coefficient
        }
        torch.save(state, save_path)
        print('save a new best checkpoints')
    else:
        print('the dice coefficient does not improve')
    
    return best_dice_one_epoch
        


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes):
    total_loss = 0
    total_f_score = 0

    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
            ce_focal_loss = loss
            if dice_loss:
                main_dice = Dice_Loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'CE/Focal_loss': ce_focal_loss.item(),
                                'dice_loss': main_dice.item(),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    loss_history.append_loss(total_loss / (epoch_step + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (total_loss / (epoch_step + 1)))
    torch.save(model.state_dict(), '../autodl-tmp/weights/ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / (epoch_step + 1)))
