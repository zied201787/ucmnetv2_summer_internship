import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils_ucm import save_imgs


from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torch.cuda.amp import autocast

def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None,
                    epoch_num=1
                    ):
    '''
    Train model for one epoch and return average loss + metrics
    '''
    model.train()
    loss_list = []

    preds_all = []
    targets_all = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images = images.cuda(non_blocking=True).float()
        targets = targets.cuda(non_blocking=True).float()

        if config.amp:
            with autocast():
                gt_pre, out = model(images)
                loss, *_ = criterion(gt_pre, out, targets, epoch, config.epochs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            gt_pre, out = model(images)
            loss, *_ = criterion(gt_pre, out, targets, epoch, epoch_num)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        # Collect predictions and targets for metrics
        out = torch.sigmoid(out)
        preds_all.append(out.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

        if iter % config.print_interval == 0:
            now_lr = optimizer.state_dict()['param_groups'][0]['lr']
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr:.6f}'
            print(log_info)
            logger.info(log_info)

    scheduler.step()

    # === Compute metrics ===
    preds_all = np.concatenate(preds_all).astype(np.uint8) > 0.5
    targets_all = np.concatenate(targets_all).astype(np.uint8)

    preds_flat = preds_all.flatten()
    targets_flat = targets_all.flatten()

    tn, fp, fn, tp = confusion_matrix(targets_flat, preds_flat, labels=[0,1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    avg_loss = np.mean(loss_list)

    logger.info(f'[Train] Epoch {epoch} | Loss: {avg_loss:.4f} | Dice: {dice:.4f} | Acc: {accuracy:.4f} | SE: {sensitivity:.4f} | SP: {specificity:.4f}')
    print(f"[Train] Epoch {epoch} | Loss: {avg_loss:.4f} | Dice: {dice:.4f} | Acc: {accuracy:.4f} | SE: {sensitivity:.4f} | SP: {specificity:.4f}")

    return avg_loss, {
        'f1': dice,
        'acc': accuracy,
        'se': sensitivity,
        'sp': specificity
    }



def val_one_epoch(test_loader, model, criterion, epoch, logger, config, epoch_num=1):
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, out = model(img)
            loss, loss1, loss2, loss3, loss4, loss5 = criterion(gt_pre, out, msk, epoch, epoch_num)

            out = torch.sigmoid(out)
            loss_list.append(loss.item())

            gts.append(msk.squeeze(1).cpu().detach().numpy())

            if type(out) is tuple:
                out = out[0]

            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    # Thresholding
    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    # Confusion matrix
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    f1_or_dsc = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    miou = TP / (TP + FP + FN + 1e-6)

    log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, ' \
               f'specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, confusion_matrix: {confusion.tolist()}'
    print(log_info)
    logger.info(log_info)
    print(accuracy, miou, sensitivity, specificity)

    return np.mean(loss_list), {
        'miou': miou,
        'f1': f1_or_dsc,
        'acc': accuracy,
        'sp': specificity,
        'se': sensitivity,
        'confusion': confusion
    }


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None, epoch=1, epoch_num=1):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            gt_pre, out = model(img)

            loss, loss1, loss2, los3, loss4, loss5 = criterion(gt_pre, out, msk, 1, 1)
            out = torch.sigmoid(out)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list), {
        'miou': miou,
        'f1': f1_or_dsc,
        'acc': accuracy,
        'sp': specificity,
        'se': sensitivity,
        'confusion': confusion
    }
