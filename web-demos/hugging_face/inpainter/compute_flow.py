import os
import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.modules.flow_loss_utils import flow_warp
from torchvision.transforms.functional import to_pil_image
import utils
import torch.nn.functional as F


pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)

def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    # fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).float()
    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)
    return fb_valid_fw



def _binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    # return mask.float()
    return mask.to(mask)

def img_propagation(x, flows_forward, flows_backward, mask, interpolation='bilinear', valid_map=True, only_forward=True, consistency_check=False):
    """
    x shape : [b, t, c, h, w]
    return [b, t, c, h, w]
    """

    # For backward warping
    # pred_flows_forward for backward feature propagation
    # pred_flows_backward for forward feature propagation
    b, t, c, h, w = x.shape
    feats, masks = {}, {}
    feats['input'] = [x[:, i, :, :, :] for i in range(0, t)]
    masks['input'] = [mask[:, i, :, :, :] for i in range(0, t)]

    prop_list = ['backward_1', 'forward_1']
    cache_list = ['input'] +  prop_list

    for p_i, module_name in enumerate(prop_list):
        feats[module_name] = []
        masks[module_name] = []

        if 'backward' in module_name:
            frame_idx = range(0, t)
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
            flows_for_prop = flows_forward
            flows_for_check = flows_backward
        else:
            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            flows_for_prop = flows_backward
            flows_for_check = flows_forward

        for i, idx in enumerate(frame_idx):
            feat_current = feats[cache_list[p_i]][idx]
            mask_current = masks[cache_list[p_i]][idx]

            if i == 0:
                feat_prop = feat_current
                mask_prop = mask_current
            else:
                flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                flow_vaild_mask = fbConsistencyCheck(flow_prop, flow_check)
                # a = flow_vaild_mask[0][0].cpu().numpy()
                # with open('tensor.txt', 'w') as f:
                #     np.savetxt(f, a, fmt='%.3f')
                # import sys
                # sys.exit(0)
                #print(feat_prop.shape, flow_prop.permute(0, 2, 3, 1).shape)
                #torch.Size([1, 3, 512, 512]) torch.Size([1, 512, 512, 2])
                # print("flow valid mask shape ", flow_vaild_mask.shape) #flow valid mask shape  torch.Size([1, 1, 512, 512])
                # print(flow_check.shape)#torch.Size([1, 2, 512, 512])
                # print(flow_prop.shape)#torch.Size([1, 2, 512, 512])
                # print("flows for prop shape ", flows_for_prop.shape) #flows for prop shape  torch.Size([1, 103, 2, 512, 512])
                feat_warped = flow_warp(feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation)


                mask_prop_valid = flow_warp(mask_prop, flow_prop.permute(0, 2, 3, 1))
                mask_prop_valid = _binary_mask(mask_prop_valid)

                union_vaild_mask = _binary_mask(mask_current*flow_vaild_mask*(1-mask_prop_valid))
                feat_prop = union_vaild_mask * feat_warped + (1-union_vaild_mask) * feat_current
                # update mask
                mask_prop = _binary_mask(mask_current*(1-(flow_vaild_mask*(1-mask_prop_valid))))
            
            feats[module_name].append(feat_prop)
            masks[module_name].append(mask_prop)

        # end for
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
            masks[module_name] = masks[module_name][::-1]

    outputs_b = torch.stack(feats['backward_1'], dim=1).view(-1, c, h, w)
    outputs_f = torch.stack(feats['forward_1'], dim=1).view(-1, c, h, w)

    masks_b = torch.stack(masks['backward_1'], dim=1)
    masks_f = torch.stack(masks['forward_1'], dim=1)
    outputs = outputs_f

    return outputs_b.view(b, -1, c, h, w), outputs_f.view(b, -1, c, h, w), \
            outputs.view(b, -1, c, h, w), masks_f



def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask


def masks_dilate(masks, flow_mask_dilates=8, mask_dilates=5):
    masks_dilated = []
    flow_masks = []
    for mask in masks:
        mask = np.array(mask.convert('L'))
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))

        if mask_dilates > 0:
            mask = scipy.ndimage.binary_dilation(mask, iterations=mask_dilates).astype(np.uint8)
        else:
            mask = binary_mask(mask).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask * 255))
    return masks_dilated, flow_masks

def read_flo_file(file_path):
    with open(file_path, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        if tag != 202021.25:
            raise ValueError(f"Invalid tag in .flo file: {tag}")
        
        width = np.fromfile(f, np.int32, count=1).item()
        height = np.fromfile(f, np.int32, count=1).item()
        data = np.fromfile(f, np.float32, count=2*width*height)
    
    flow = np.resize(data, (height, width, 2))
    return flow

# Function to read multiple .flo files and stack them into a tensor
def read_multiple_flo_files(file_paths):
    flow_list = []
    for file_path in file_paths:
        flow_np = read_flo_file(file_path)
        flow_tensor = torch.from_numpy(flow_np)
        flow_list.append(flow_tensor)
    
    # Stack all tensors along the first dimension (frames)
    flow_stack = torch.stack(flow_list, dim=0)
    return flow_stack

def resize_flow(flow, new_height, new_width):
    batch_size, num_maps, channels, old_height, old_width = flow.shape
    resized_flow = torch.zeros((batch_size, num_maps, channels, new_height, new_width), device=flow.device)

    # Resizing each flow map individually
    for i in range(num_maps):
        # Extract the individual flow map
        single_flow = flow[:, i]

        # Resizing
        resized_single_flow = F.interpolate(single_flow, size=(new_height, new_width), mode='bilinear', align_corners=True)

        # Adjusting the flow values
        resized_single_flow[:, 0, :, :] *= (new_width / old_width)
        resized_single_flow[:, 1, :, :] *= (new_height / old_height)

        resized_flow[:, i] = resized_single_flow

    return resized_flow

def get_ground_truth_flow():
    file_path = '../../Downloads/MPI-Sintel-training_extras/training/flow/bandage_2'
    import os
    l = len(os.listdir(file_path))
    l = 19
    file_paths = [f'{file_path}/frame_{i:04d}.flo' for i in range(1, l + 1)]
    flow_tensor_stack = read_multiple_flo_files(file_paths)
    return flow_tensor_stack
        
def compute_flow(frames, masks, dilate=True):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    masks_dilated, flow_masks = masks_dilate(masks, flow_mask_dilates=8, mask_dilates=5)
    if dilate == False:
        masks_dilated, flow_masks = masks_dilate(masks, flow_mask_dilates=0, mask_dilates=0)
    w, h = (512, 512)
    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

    use_half = True

    
    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    print('Loading models...')
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    video_length = frames.size(1)
    with torch.no_grad():
        # ---- compute flow ----
        print('Computing flow...')
        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        
        raft_iter = 20
        # use fp32 for RAFT
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=raft_iter)
            torch.cuda.empty_cache()


        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()

        
        # print("forward flow shape here")
        # print(gt_flows_bi[0].shape)
        # ground_truth_forward = get_ground_truth_flow()
        # print(ground_truth_forward.shape)
        # ground_truth_forward = ground_truth_forward.permute(0, 3, 1, 2).unsqueeze(0)
        # print(ground_truth_forward.shape)
        # ground_truth_forward = resize_flow(ground_truth_forward, 512, 512)
        # ground_truth_forward = ground_truth_forward.to(device='cuda:0', dtype=torch.float16)
        # print(ground_truth_forward.shape)
        # gt_flows_bi = (ground_truth_forward, gt_flows_bi[1])

        # ---- complete flow ----
        print('Completing flow...')
        pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
        pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
        torch.cuda.empty_cache()

        
        # ---- image propagation ----
        print('Propagating images...')
        masked_frames = frames * (1 - masks_dilated)
        b, t, _, _, _ = masks_dilated.size()
        _, _, prop_imgs, updated_local_masks = img_propagation(masked_frames, pred_flows_bi[0], pred_flows_bi[1], masks_dilated, 'nearest')
        updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
        updated_masks = updated_local_masks.view(b, t, 1, h, w)
        torch.cuda.empty_cache()

    updated_frames = (updated_frames + 1) / 2      
    updated_frames = updated_frames.squeeze(0)  
    updated_frames_tensor = updated_frames
    # print("before transfer frame size ", updated_frames.shape) #[t, 3, 512, 512]
    updated_frames = [to_pil_image(updated_frame) for updated_frame in updated_frames]
    updated_masks = updated_masks.squeeze(0)
    updated_masks_tensor = updated_masks
    # print("before transfer mask size ", updated_masks.shape) #[t, 1, 512, 512]
    updated_masks = [to_pil_image(updated_mask) for updated_mask in updated_masks]

    # utils.flow_util.visualize_flow(np.array(updated_frames[0]), gt_flows_bi[0], 'data/output1/boat', 'forward')
    # utils.flow_util.visualize_flow(np.array(updated_frames[0]), gt_flows_bi[1], 'data/output1/boat', 'backward')

    return updated_frames, updated_masks, pred_flows_bi, updated_frames_tensor, updated_masks_tensor
