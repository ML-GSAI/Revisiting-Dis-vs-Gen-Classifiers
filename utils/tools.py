import logging
import os
import time
import random
import numpy as np
import torch
from scipy import interpolate

from tqdm import tqdm

def get_console_file_logger(name, level=logging.INFO, logdir='./baseline'):
    logger = logging.Logger(name)
    logger.setLevel(level=level)
    logger.handlers = []
    BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(level=level)

    fhlr = logging.FileHandler(os.path.join(logdir, str(time.time()) + '.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    return logger

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.enabled = True

def get_error_n(n, batch_size=10):
    m = 100000
    iter = m // batch_size
    error = 0
    for i in tqdm(range(iter)):
        x = np.random.multivariate_normal(mean=n*[0], cov=np.eye(n), size=batch_size)
        error += np.sum(np.sum(x[:,:n//2+1],1) / np.sqrt(n) + np.sum(x[:,n//2+1:], 1) < -(n+1)/2)
    return error / m

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def load_pretrained(model, args):
    print(f">>>>>>>>>> Linear eval from {args.backbone_path} ..........")
    checkpoint = torch.load(args.backbone_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    
    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        print('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        print('Detect non-pre-trained model, pass without doing anything.')

    assert args.backbone == 'simmim'
    print(f">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
    checkpoint = remap_pretrained_keys_vit(model, checkpoint_model)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    del checkpoint
    torch.cuda.empty_cache()
    print(f">>>>>>>>>> loaded successfully from '{args.backbone_path}'")

def remap_pretrained_keys_vit(model, checkpoint_model):
    # Duplicate shared rel_pos_bias to each layer
    if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        print("Expand the shared relative position embedding to each transformer block.")
    num_layers = model.get_num_layers()
    rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
    for i in range(num_layers):
        checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
    checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print("Position interpolate for %s from %dx%d to %dx%d" % (key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions = %s" % str(x))
                print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias
    
    return checkpoint_model