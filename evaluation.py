import torch

# SR: 分割结果, 预测
# GT: 地面真值, 标签

def Pixel_Accuracy(SR, GT):
    SR = SR.flatten()
    GT = GT.flatten()

    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0)
    PA = float(corr) / float(tensor_size)
    return PA

def Recall(SR, GT):
    SR = SR.flatten()
    GT = GT.flatten()

    TP = torch.sum(torch.mul(SR, GT))
    FN = torch.sum((GT == 1) & (SR == 0))
    recall = float(torch.sum(TP)) / (float(torch.sum(TP+FN)) + 1e-6)
    return recall

#返回GT中值为1在SR中值为1的像素点的个数与SR中值为1的像素点的个数的比值
def Precision(SR, GT):
    SR = SR.flatten()
    GT = GT.flatten()
    TP = torch.sum(torch.mul(SR, GT))
    FP = torch.sum(torch.sum((GT == 0) & (SR == 1)))
    Precision = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return Precision

def F1(SR, GT):
    recall = Recall(SR, GT)
    precision = Precision(SR, GT)
    F1 = 2 * recall * precision / (recall + precision +1e-6)
    return F1

#基本等同将SR和GT拼接然后返回其中不重复的元素由小到大组成的张量和张量的长度
def union_classes(SR, GT):
    eval_cl, _ = extract_classes(SR)
    gt_cl, _ = extract_classes(GT)

    cl = torch.unique(torch.cat([eval_cl, gt_cl]).view(-1))
    n_cl = len(cl)
    return cl, n_cl

#返回GT中不重复的元素由小到大组成的张量和张量的长度
def extract_classes(GT):
    cl = torch.unique(GT)#unique()挑选出GT中不重复的元素，并按照顺序排列，然后返回张量
    n_cl = len(cl)
    return cl, n_cl

#返回张量的第0维和第1维的长度
def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise
    return height, width

#返回一个三维张量（对于每一个像素点的和为1）
def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = torch.zeros((n_cl, h, w))
    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c #先计算等号，再赋值
    return masks

#对于预测和标签同时调用extract_masks()，返回两个三维张量
def extract_both_masks(SR, GT, cl, n_cl):
    eval_mask = extract_masks(SR, cl, n_cl)
    gt_mask = extract_masks(GT, cl, n_cl)
    return eval_mask, gt_mask

#返回Miou值
def mean_IU(SR, GT):
    cl, n_cl = union_classes(SR, GT)
    _, gt_n_cl = extract_classes(GT)
    eval_mask, gt_mask = extract_both_masks(SR, GT, cl, n_cl)

    IU = torch.FloatTensor(list([0]) * n_cl)#返回一个长度为n_cl的全零float型张量
    #将eval_mask和gt_mask的每一层单独拿出来
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (torch.sum(curr_eval_mask) == 0) or (torch.sum(curr_gt_mask) == 0):
            continue

        n_ii = torch.sum((curr_eval_mask == 1) & (curr_gt_mask == 1))#计算预测和标签中同时为1的个数
        t_i = torch.sum(curr_gt_mask)#真值中1的个数
        n_ij = torch.sum(curr_eval_mask)#预测中1的个数

        IU[i] = n_ii / (t_i + n_ij - n_ii)#在第i个图层的iou值

    miou = torch.sum(IU) / gt_n_cl#计算Miou值
    return miou

#总体准确度
def Overall_Accuracy(SR, GT):
    SR = SR.flatten()
    GT = GT.flatten()
    TP = torch.sum(torch.mul(SR, GT))
    FP = torch.sum((SR == 0) & (GT == 1))
    TN = torch.sum((SR == 0) & (GT == 0))
    FN = torch.sum((SR == 1) & (GT == 0))
    OA = float(torch.sum(TP+TN)) / (float(torch.sum(TP + TN + FN + FP)) + 1e-6)
    return OA

def Kappa(SR, GT):
    SR = SR.flatten()
    GT = GT.flatten()
    TP = torch.sum(torch.mul(SR, GT))
    FP = torch.sum((SR == 0) & (GT == 1))
    TN = torch.sum((SR == 0) & (GT == 0))
    FN = torch.sum((SR == 1) & (GT == 0))
    Po = Overall_Accuracy(SR, GT)
    TNFN = TN + FN
    TNFP = TN + FP
    FPTP = FP + TP
    FNTP = FN + TP
    Pe =float(((TNFN*TNFP)+(FPTP*FNTP)))/float((TP+FP+TN+FN)*(TP+FP+TN+FN))
    kappa = float(Po-Pe) / (float(1-Pe) + 1e-6)
    return kappa

if __name__ == "__main__":
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import numpy as np

    SR = torch.IntTensor([1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]).view(5, 5)
    GT = torch.IntTensor([1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]).view(5, 5)


    PA = Pixel_Accuracy(SR, GT)
    recall = Recall(SR, GT)
    precision = Precision(SR, GT)
    f1 = F1(SR, GT)
    # OA = Overall_Accuracy(SR, GT)
    # Kappa = Kappa(SR, GT)

    print(union_classes(SR, GT))
    print(segm_size(SR))
    cl, n_cl=union_classes(SR, GT)
    print(extract_both_masks(SR, GT, cl, n_cl))
    print(mean_IU(SR, GT))
    print(Overall_Accuracy(SR, GT))
    print(Kappa(SR, GT))
    print('PA           code: {:.3f} | Sklearn: {:.3f}'.format(PA, accuracy_score(GT.flatten(), SR.flatten())))
    print('Recall       code: {:.3f} | Sklearn: {:.3f}'.format(recall, recall_score(GT.flatten(), SR.flatten())))
    print('Precision    code: {:.3f} | Sklearn: {:.3f}'.format(precision, precision_score(GT.flatten(), SR.flatten())))
    print('F1           code: {:.3f} | Sklearn: {:.3f}'.format(f1, f1_score(GT.flatten(), SR.flatten())))
    print("************测试结束**************")



