import logging
import os
import sys
import os.path as osp
def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            # 从 save_dir 路径中提取最后一个目录名作为日志文件名
            # 例如: ./logs/alpha_v2_market -> alpha_v2_market.txt
            dir_basename = osp.basename(save_dir.rstrip('/\\'))
            if dir_basename:
                log_filename = f"{dir_basename}.txt"
            else:
                log_filename = "train_log.txt"
            fh = logging.FileHandler(os.path.join(save_dir, log_filename), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
