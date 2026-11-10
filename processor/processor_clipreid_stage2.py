import csv
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.attention_maps import get_attention_run_dir, save_attention_batch
from utils.grad_cam import compute_reid_grad_cam, create_reid_grad_cam
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from utils.checkpoint import (
    make_stage2_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


STAGE2_METRIC_FIELDS = (
    "epoch",
    "mAP_percent",
    "rank1_percent",
    "rank5_percent",
    "stage2_epoch_training_seconds",
    "stage2_cumulative_training_seconds",
    "stage2_peak_gpu_memory_mb",
)


def append_stage2_epoch_metrics(
    output_dir,
    epoch,
    mAP,
    cmc,
    epoch_training_seconds,
    cumulative_training_seconds,
    peak_gpu_memory_mb,
):
    metrics_path = os.path.join(output_dir, "stage2_epoch_metrics.csv")
    file_exists = os.path.isfile(metrics_path) and os.path.getsize(metrics_path) > 0
    with open(metrics_path, "a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=STAGE2_METRIC_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "mAP_percent": round(float(mAP) * 100, 4),
                "rank1_percent": round(float(cmc[0]) * 100, 4),
                "rank5_percent": round(float(cmc[4]) * 100, 4),
                "stage2_epoch_training_seconds": round(
                    epoch_training_seconds, 3
                ),
                "stage2_cumulative_training_seconds": round(
                    cumulative_training_seconds, 3
                ),
                "stage2_peak_gpu_memory_mb": round(peak_gpu_memory_mb, 2),
            }
        )
        metrics_file.flush()


def _random_pool_i2t_logits(
    image_features, text_features, targets, num_classes, num_negatives
):
    positive_ids = torch.unique(targets, sorted=True)
    negative_mask = torch.ones(
        num_classes, dtype=torch.bool, device=targets.device
    )
    negative_mask[positive_ids] = False
    negative_ids = torch.arange(num_classes, device=targets.device)[negative_mask]

    sample_count = min(num_negatives, negative_ids.numel())
    if sample_count:
        permutation = torch.randperm(negative_ids.numel(), device=targets.device)
        negative_ids = negative_ids[permutation[:sample_count]]

    candidate_ids = torch.cat((positive_ids, negative_ids))
    logits = image_features @ text_features[candidate_ids].t()
    target_indices = torch.searchsorted(positive_ids, targets)
    return logits, target_indices, candidate_ids


def _random_k_i2t_logits(
    image_features, text_features, targets, num_classes, num_negatives
):
    num_negatives = min(num_negatives, num_classes - 1)
    class_permutation = torch.randperm(num_classes, device=targets.device)
    class_positions = torch.empty_like(class_permutation)
    class_positions[class_permutation] = torch.arange(
        num_classes, device=targets.device
    )

    offsets = torch.arange(num_negatives, device=targets.device)
    negative_positions = (
        class_positions[targets].unsqueeze(1) + 1 + offsets
    ) % num_classes
    negative_ids = class_permutation[negative_positions]

    positive_features = text_features[targets]
    negative_features = text_features[negative_ids]
    positive_logits = (image_features * positive_features).sum(dim=1, keepdim=True)
    negative_logits = torch.einsum(
        "bd,bkd->bk", image_features, negative_features
    )
    logits = torch.cat((positive_logits, negative_logits), dim=1)
    candidate_ids = torch.cat((targets.unsqueeze(1), negative_ids), dim=1)
    target_indices = torch.zeros_like(targets)
    return logits, target_indices, candidate_ids


def do_train_stage2(
    cfg,
    model,
    center_criterion,
    train_loader_stage2,
    val_loader,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    num_query,
    local_rank,
    resume_state=None,
):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    start_epoch = 1
    cumulative_training_seconds = 0.0
    stage2_peak_memory_mb = 0.0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        optimizer_center.load_state_dict(resume_state["optimizer_center"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        start_epoch = resume_state["epoch"] + 1
        cumulative_training_seconds = resume_state["stage2_metrics"][
            "cumulative_training_seconds"
        ]
        stage2_peak_memory_mb = resume_state["stage2_metrics"][
            "peak_gpu_memory_mb"
        ]
        if start_epoch > epochs:
            raise ValueError(
                "Checkpoint completed epoch {}; set SOLVER.STAGE2.MAX_EPOCHS "
                "to a value greater than {} to continue training.".format(
                    resume_state["epoch"], resume_state["epoch"]
                )
            )
    
    from datetime import timedelta

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    if resume_state is not None:
        restore_rng_state(resume_state["rng_state"])

    def save_stage2_state(epoch, checkpoint_path):
        checkpoint = make_stage2_checkpoint(
            epoch,
            model,
            center_criterion,
            optimizer,
            optimizer_center,
            scheduler,
            scaler,
            {
                "cumulative_training_seconds": cumulative_training_seconds,
                "peak_gpu_memory_mb": stage2_peak_memory_mb,
            },
        )
        save_checkpoint(checkpoint, checkpoint_path)
        logger.info("Saved Stage 2 checkpoint to {}".format(checkpoint_path))

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        torch.cuda.reset_peak_memory_stats(local_rank)
        training_epoch_start = time.perf_counter()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                if cfg.I2T_RANDOM_K.ENABLED:
                    logits, i2t_target, candidate_ids = _random_k_i2t_logits(
                        image_features,
                        text_features,
                        target,
                        num_classes,
                        cfg.I2T_RANDOM_K.NUMBER_K,
                    )
                elif cfg.I2T_RANDOM_POOL.ENABLED:
                    logits, i2t_target, candidate_ids = _random_pool_i2t_logits(
                        image_features,
                        text_features,
                        target,
                        num_classes,
                        cfg.I2T_RANDOM_POOL.NUM_NEGATIVES,
                    )
                else:
                    logits = image_features @ text_features.t()
                    i2t_target = target
                    candidate_ids = torch.arange(num_classes, device=target.device)
                loss = loss_fn(
                    score, feat, target, target_cam, logits, i2t_target
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            predicted_indices = logits.max(1)[1]
            if candidate_ids.ndim == 1:
                predicted_ids = candidate_ids[predicted_indices]
            else:
                predicted_ids = candidate_ids.gather(
                    1, predicted_indices.unsqueeze(1)
                ).squeeze(1)
            acc = (predicted_ids == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        torch.cuda.synchronize(local_rank)
        epoch_training_seconds = time.perf_counter() - training_epoch_start
        cumulative_training_seconds += epoch_training_seconds
        epoch_peak_memory_mb = torch.cuda.max_memory_allocated(local_rank) / (
            1024 ** 2
        )
        stage2_peak_memory_mb = max(stage2_peak_memory_mb, epoch_peak_memory_mb)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        epoch_metrics = None
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    epoch_metrics = (mAP, cmc)
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                epoch_metrics = (mAP, cmc)
                torch.cuda.empty_cache()

        if epoch_metrics is not None:
            append_stage2_epoch_metrics(
                cfg.OUTPUT_DIR,
                epoch,
                epoch_metrics[0],
                epoch_metrics[1],
                epoch_training_seconds,
                cumulative_training_seconds,
                stage2_peak_memory_mb,
            )
            logger.info(
                "Saved Stage 2 epoch metrics to {}".format(
                    os.path.join(cfg.OUTPUT_DIR, "stage2_epoch_metrics.csv")
                )
            )

        if epoch % checkpoint_period == 0:
            if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
                save_stage2_state(
                    epoch,
                    os.path.join(
                        cfg.OUTPUT_DIR,
                        cfg.MODEL.NAME + "_{}.pth".format(epoch),
                    ),
                )

    if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
        save_stage2_state(
            epochs,
            os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_final.pth"),
        )

    total_training_time = timedelta(seconds=cumulative_training_seconds)
    logger.info("Stage2 training time: {}".format(total_training_time))
    logger.info(
        "Stage2 peak GPU memory allocated: {:.2f} MB".format(
            stage2_peak_memory_mb
        )
    )
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 occlusion_ratio=0.0):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    attention_enabled = cfg.TEST.ATTENTION.ENABLED
    attention_image_limit = int(cfg.TEST.ATTENTION.NUM_IMAGES)
    if attention_image_limit < 0:
        raise ValueError("TEST.ATTENTION.NUM_IMAGES must not be negative.")
    remaining_attention_images = attention_image_limit or None
    evaluator = R1_mAP_eval(
        num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM
    )
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1 and not attention_enabled:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        elif torch.cuda.device_count() > 1:
            logger.info("Attention-map capture is enabled; using one GPU.")
        model.to(device)

    model.eval()
    img_path_list = []
    if attention_enabled:
        index_path = get_attention_run_dir(cfg, occlusion_ratio) / "index.jsonl"
        if index_path.exists():
            index_path.unlink()

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
        if attention_enabled and remaining_attention_images != 0:
            capture_count = (
                len(imgpath)
                if remaining_attention_images is None
                else min(len(imgpath), remaining_attention_images)
            )
            cam_labels = camids[:capture_count] if camids is not None else None
            view_labels = (
                target_view[:capture_count] if target_view is not None else None
            )
            cam = create_reid_grad_cam(model, cam_labels, view_labels)
            try:
                with torch.enable_grad():
                    attention_maps = compute_reid_grad_cam(
                        cam, img[:capture_count]
                    )
            finally:
                cam.activations_and_grads.release()
            save_attention_batch(
                cfg,
                imgpath[:capture_count],
                img[:capture_count],
                attention_maps,
                occlusion_ratio,
            )
            if remaining_attention_images is not None:
                remaining_attention_images -= capture_count

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]