import torch.nn.functional as F

def custom_loss(output, target):
    conf_output, conf_target = output[:, :, 0], target[:, :, 0]
    coor_output, coor_target = output[:, :, 1:5], target[:, :, 1:5]
    class_output, class_target = output[:, :, 5:], target[:, :, 5:]
    mask = conf_target == 1
    conf_loss = F.binary_cross_entropy(conf_output, conf_target)
    coor_loss = F.mse_loss(coor_output[mask], coor_target[mask])
    class_output = class_output[mask]
    class_target = class_target[mask].squeeze(-1).long()
    class_loss = F.cross_entropy(class_output, class_target)
    total_loss = (conf_loss + coor_loss + class_loss) / 3
    return total_loss