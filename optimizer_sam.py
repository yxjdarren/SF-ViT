# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim
from sam.sam import SAM


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.

    SAM simultaneously minimizes loss value and loss sharpness.
    In particular, it seeks parameters that lie in neighborhoods having uniformly low loss.
    SAM improves model generalization and yields SoTA performance for several datasets.
    Additionally, it provides robustness to label noise on par with that provided by SoTA procedures
    that specifically target learning with noisy labels.

    SAM算法需要前向-后向计算两次梯度，因此需要传入一个闭包去允许重新计算模型
    这个闭包应当清空梯度、计算损失、然后返回
        example:

        for input, target in dataset:
            def closure():
                optimizer.zero_gard()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()

                return loss
            optimizer.step(closure)
    """

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer_sam = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD
        optimizer_sam = SAM(parameters, optimizer, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY,
                            momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW
        optimizer_sam = SAM(parameters, optimizer, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY,
                            eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS)

    return optimizer_sam


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
