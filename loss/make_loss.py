# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def _topk_i2t_loss(logits, targets, k, epsilon):
    negative_logits = logits.clone()
    negative_logits.scatter_(1, targets.unsqueeze(1), float("-inf"))
    k = min(k, logits.size(1) - 1)
    topk_negative_logits = negative_logits.topk(k, dim=1).values
    positive_logits = logits.gather(1, targets.unsqueeze(1))
    candidate_logits = torch.cat((positive_logits, topk_negative_logits), dim=1)

    return _label_smoothed_i2t_loss(
        candidate_logits, torch.zeros_like(targets), epsilon
    )


def _label_smoothed_i2t_loss(logits, targets, epsilon):
    log_probs = F.log_softmax(logits, dim=1)
    nll_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    smooth_loss = -log_probs.mean(dim=1)
    return ((1 - epsilon) * nll_loss + epsilon * smooth_loss).mean()


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    topk_i2t_enabled = cfg.I2T_TOPK.ENABLED
    random_pool_i2t_enabled = cfg.I2T_RANDOM_POOL.ENABLED
    random_k_i2t_enabled = cfg.I2T_RANDOM_K.ENABLED
    if random_k_i2t_enabled and topk_i2t_enabled:
        raise ValueError("Enable either I2T_TOPK or I2T_RANDOM_K, not both")
    if random_k_i2t_enabled and random_pool_i2t_enabled:
        raise ValueError("I2T_RANDOM_K and I2T_RANDOM_POOL cannot both be enabled")

    if random_pool_i2t_enabled:
        if cfg.I2T_RANDOM_POOL.NUM_NEGATIVES < 1:
            raise ValueError("I2T_RANDOM_POOL.NUM_NEGATIVES must be at least 1")
        if num_classes < 2:
            raise ValueError("Random-pool I2T loss requires at least two classes")
        print(
            "using a random pool of up to {} negative classes for I2T loss".format(
                cfg.I2T_RANDOM_POOL.NUM_NEGATIVES
            )
        )

    if random_k_i2t_enabled:
        if cfg.I2T_RANDOM_K.NUMBER_K < 1:
            raise ValueError("I2T_RANDOM_K.NUMBER_K must be at least 1")
        if num_classes < 2:
            raise ValueError("Random-K I2T loss requires at least two classes")
        random_k_i2t_k = min(cfg.I2T_RANDOM_K.NUMBER_K, num_classes - 1)
        print("using {} random negative classes for I2T loss".format(random_k_i2t_k))

    if topk_i2t_enabled:
        if cfg.I2T_TOPK.NUMBER_K < 1:
            raise ValueError("I2T_TOPK.NUMBER_K must be at least 1")
        if num_classes < 2:
            raise ValueError("Top-K I2T loss requires at least two classes")
        topk_i2t_k = min(cfg.I2T_TOPK.NUMBER_K, num_classes - 1)
        print("using top-{} negative logits for I2T loss".format(topk_i2t_k))

    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(
            score, feat, target, target_cam, i2tscore=None, i2t_target=None
        ):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    if i2tscore is not None:
                        i2t_target = target if i2t_target is None else i2t_target
                        if topk_i2t_enabled:
                            I2TLOSS = _topk_i2t_loss(
                                i2tscore, i2t_target, topk_i2t_k, xent.epsilon
                            )
                        elif random_pool_i2t_enabled or random_k_i2t_enabled:
                            I2TLOSS = _label_smoothed_i2t_loss(
                                i2tscore, i2t_target, xent.epsilon
                            )
                        else:
                            I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        
                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore is not None:
                        i2t_target = target if i2t_target is None else i2t_target
                        if topk_i2t_enabled:
                            I2TLOSS = _topk_i2t_loss(
                                i2tscore, i2t_target, topk_i2t_k, 0
                            )
                        else:
                            I2TLOSS = F.cross_entropy(i2tscore, i2t_target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
