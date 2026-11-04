import os
from config import cfg
import random
import argparse
import numpy as np
import torch
from datasets.make_dataloader_clipreid import make_dataloader
from datasets.preprocessing import build_test_transform
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    random.seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    if cfg.TEST.OCCLUSION.ENABLED:
        logger.info("Evaluation condition: baseline; occlusion_ratio=0")
        val_loader.dataset.transform = build_test_transform(
            cfg.INPUT.SIZE_TEST,
            cfg.INPUT.PIXEL_MEAN,
            cfg.INPUT.PIXEL_STD,
        )
        do_inference(cfg, model, val_loader, num_query, 0.0)

        for ratio in cfg.TEST.OCCLUSION.RATIOS:
            if not 0 < ratio <= 1:
                raise ValueError(
                    "Occlusion ratios must be greater than 0 and at most 1."
                )
            random.seed(cfg.SOLVER.SEED)
            np.random.seed(cfg.SOLVER.SEED)
            torch.manual_seed(cfg.SOLVER.SEED)
            torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
            logger.info(
                "Evaluation condition: random_patch; occlusion_ratio={}".format(
                    ratio
                )
            )
            logger.info("Testing with random patch occlusion: %.0f%%", ratio * 100)
            val_loader.dataset.transform = build_test_transform(
                cfg.INPUT.SIZE_TEST,
                cfg.INPUT.PIXEL_MEAN,
                cfg.INPUT.PIXEL_STD,
                ratio if ratio > 0 else None,
            )
            do_inference(cfg, model, val_loader, num_query, ratio)
    elif cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5, mAP = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
                all_mAP = mAP
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
                all_mAP = all_mAP + mAP

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, mAP, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}, sum_mAP {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0, all_mAP.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
