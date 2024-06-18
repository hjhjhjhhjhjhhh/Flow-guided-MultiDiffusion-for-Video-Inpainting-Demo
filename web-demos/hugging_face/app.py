import sys
sys.path.append("../../")

import os
import json
import time
import psutil
import argparse

import cv2
import torch
import torchvision
import numpy as np
import gradio as gr
from PIL import Image

from tools.painter import mask_painter
from track_anything import TrackingAnything

from model.misc import get_device
from utils.download_util import load_file_from_url


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000, help="only useful when running gradio applications")  
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()
    
    if not args.device:
        args.device = str(get_device())

    return args 

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt

# extract frames from upload video
def get_frames_from_video(video_input, video_state, task_type):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [("[Must Do]", "Click image"), (": Video uploaded! Try to click the image shown in step2 to add masks.\n", None)]
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps
        }
    video_info = "Video Name: {},\nFPS: {},\nTotal Frames: {},\nImage Size:{}".format(video_state["video_name"], round(video_state["fps"], 0), len(frames), image_size)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    if task_type == "object_removal":
        return video_state, video_info, video_state["origin_images"][0], video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), \
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=False),\
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=True), gr.update(visible=True), \
                            gr.update(visible=True), gr.update(visible=True, choices=[], value=[]), \
                            gr.update(visible=True, value=operation_log), gr.update(visible=True, value=operation_log)
    return video_state, video_info, video_state["origin_images"][0], video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=False, choices=[], value=[]), \
                        gr.update(visible=True, value=operation_log), gr.update(visible=True, value=operation_log)
    

# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, mask_dropdown):

    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    operation_log = [("",""), ("Select tracking start frame {}. Try to click the image to add masks for tracking.".format(image_selection_slider),"Normal")]

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log, operation_log

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Select tracking finish frame {}.Try to click the image to add masks for tracking.".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log, operation_log

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("[Must Do]", "Add mask"), (": add the current displayed mask for video segmentation.\n", None),
                     ("[Optional]", "Remove mask"), (": remove all added masks.\n", None),
                     ("[Optional]", "Clear clicks"), (": clear current displayed mask.\n", None),
                     ("[Optional]", "Click image"), (": Try to click the image shown in step2 if you want to generate more masks.\n", None)]
    return painted_image, video_state, interactive_state, operation_log, operation_log

def add_multi_mask(video_state, interactive_state, mask_dropdown, task_type, mask_frame):
    temp_save = mask_frame
    if task_type == 'object_removal':
        try:
            mask = video_state["masks"][video_state["select_frame_number"]]
            interactive_state["multi_mask"]["masks"].append(mask)
            interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
            mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
            select_frame, _, _ = show_mask(video_state, interactive_state, mask_dropdown)
            print("type select frame ", type(select_frame))
            print(select_frame.shape)
            
            operation_log = [("",""),("Added a mask, use the mask select for target tracking or inpainting.","Normal")]
        except:
            operation_log = [("Please click the image in step2 to generate masks.", "Error"), ("","")]
        return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log, operation_log, None, temp_save
    
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        operation_log = [("",""),("Added a mask, use the mask select for target tracking or inpainting.","Normal")]
        threshold = 0
        mask_output = np.asarray(mask_frame['layers'][0])
        if task_type == "watermark_removal(upload mask)":
            mask_output = np.asarray(mask_frame['background'])
        binary_mask = (mask_output[:, :, 2] > threshold).astype(np.uint8)
        mask_frame['layers'][0] = binary_mask
        mask_expand = np.expand_dims(binary_mask, axis=-1)
        mask_frame['composite'] = mask_frame['background'] * (1 - mask_expand)
        select_frame = Image.fromarray(mask_frame['composite']).convert('RGB')
        select_frame = np.array(select_frame)

        return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log, operation_log, binary_mask * 255, temp_save
    except:
        operation_log = [("Please click the image in step2 to generate masks.", "Error"), ("","")]

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Cleared points history and refresh the image.","Normal")]
    return template_frame, click_state, operation_log, operation_log

def remove_multi_mask(interactive_state, mask_dropdown, temp_save, template_frame1):
    temp_save['composite'] = temp_save['background']
    template_frame1['layers'][0] = Image.new('RGBA', template_frame1['layers'][0].size, (0, 0, 0, 0))
    temp_save['layers'][0] = Image.new('RGBA', temp_save['layers'][0].size, (0, 0, 0, 0))
    template_frame1['composite'] = temp_save['composite']
    # temp_save['background'].save('background.png')
    # temp_save['layers'][0].save('layers.png')
    # temp_save['composite'].save('composite.png')
    # template_frame1['background'].save('background1.png')
    # template_frame1['layers'][0].save('layers1.png')
    # template_frame1['composite'].save('composite1.png')
    # interactive_state["multi_mask"]["mask_names"]= []
    # interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all masks. Try to add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log, operation_log, temp_save

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
    
    operation_log = [("",""), ("Added masks {}. If you want to do the inpainting with current masks, please go to step3, and click the Tracking button first and then Inpainting button.".format(mask_dropdown),"Normal")]
    return select_frame, operation_log, operation_log

# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown, task_type, mask_frame):
    operation_log = [("",""), ("Tracking finished! Try to click the Inpainting button to get the inpainting result.","Normal")]
    model.cutie.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if task_type == 'object_removal':
        if len(np.unique(template_mask))==1:
            template_mask[0][0]=1
            operation_log = [("Please add at least one mask to track by clicking the image in step2.","Error"), ("","")]
            # return video_output, video_state, interactive_state, operation_error
        masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
        # clear GPU memory
        model.cutie.clear_memory()
    else:
        masks = []
        logits = []
        painted_images = []
        mask_frame = mask_frame.resize((following_frames[0].shape[1], following_frames[0].shape[0]))
        mask_frame = np.array(mask_frame)
        threshold = 128
        binary_mask = (mask_frame[:, :, 2] > threshold).astype(np.uint8)
        mask_frame = np.expand_dims(binary_mask, axis=-1)

        for i in range(len(following_frames)):
            masks.append(mask_frame)
            painted_images.append(following_frames[i] * (1 - mask_frame))
        
        
    if interactive_state["track_end_number"]: 
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
        video_state["painted_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    video_output = generate_video_from_frames(video_state["painted_images"], output_path="./result/track/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video
    interactive_state["inference_times"] += 1
    
    print("For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(interactive_state["inference_times"], 
                                                                                                                                           interactive_state["positive_click_times"]+interactive_state["negative_click_times"],
                                                                                                                                           interactive_state["positive_click_times"],
                                                                                                                                        interactive_state["negative_click_times"]))

    #### shanggao code for mask save
    if interactive_state["mask_save"]:
        if not os.path.exists('./result/mask/{}'.format(video_state["video_name"].split('.')[0])):
            os.makedirs('./result/mask/{}'.format(video_state["video_name"].split('.')[0]))
        i = 0
        print("save mask")
        for mask in video_state["masks"]:
            np.save(os.path.join('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), '{:05d}.npy'.format(i)), mask)
            i+=1
        # save_mask(video_state["masks"], video_state["video_name"])
    #### shanggao code for mask save
    return video_output, video_state, interactive_state, operation_log, operation_log

# inpaint 
def inpaint_video(video_state, mask_dropdown, prompt, negative_prompt):
    operation_log = [("",""), ("Inpainting finished!","Normal")]

    frames = np.asarray(video_state["origin_images"])
    fps = video_state["fps"]
    inpaint_masks = np.asarray(video_state["masks"])
    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    mask_dropdown.sort()
    # convert mask_dropdown to mask numbers
    inpaint_mask_numbers = [int(mask_dropdown[i].split("_")[1]) for i in range(len(mask_dropdown))]
    # interate through all masks and remove the masks that are not in mask_dropdown
    unique_masks = np.unique(inpaint_masks)
    num_masks = len(unique_masks) - 1
    for i in range(1, num_masks + 1):
        if i in inpaint_mask_numbers:
            continue
        inpaint_masks[inpaint_masks==i] = 0
    
    # inpaint for videos
    inpainted_frames = model.baseinpainter.run(frames, inpaint_masks, prompt, negative_prompt)

    video_output = generate_video_from_frames(inpainted_frames, output_path="./result/inpaint/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video

    return video_output, operation_log, operation_log

# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path

def restart():
    operation_log = [("",""), ("Try to upload your video and click the Get video info button to get started!", "Normal")]
    return {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
        }, {
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": args.mask_save,
            "multi_mask": {
                "mask_names": [],
                "masks": []
            },
            "track_end_number": None,
        }, [[],[]], None, None, None, \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),\
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "", \
        gr.update(visible=True, value=operation_log), gr.update(visible=False, value=operation_log)


# args, defined in track_anything.py
args = parse_augment()
pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
checkpoint_fodler = os.path.join('..', '..', 'weights')

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_fodler)
cutie_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'cutie-base-mega.pth'), checkpoint_fodler)
propainter_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'ProPainter.pth'), checkpoint_fodler)
raft_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'raft-things.pth'), checkpoint_fodler)
flow_completion_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), checkpoint_fodler)

# initialize sam, cutie, propainter models
model = TrackingAnything(sam_checkpoint, cutie_checkpoint, propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, args)


title = r"""<h1 align="center">Flow-Guided Multidiffusion for Video Inpainting</h1>"""

description = r"""
"""

article = r"""
---
📝 **Thanks**
<br>
Our demo code is modified from <a href='https://github.com/sczhou/ProPainter' target='_blank' style='color: white;'>Propainter</a> and <a href='https://github.com/aiiu-lab/MeDM' target='_blank' style='color: white;'>MeDM</a>. We greatly appreciate their work.

📋 **License**
<br>
This project is licensed under <a rel="license" href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE">S-Lab License 1.0</a>. 
Redistribution and use for non-commercial purposes should follow this license.

"""
css = """
.gradio-container {width: 85% !important}
.gr-monochrome-group {border-radius: 5px !important; border: revert-layer !important; border-width: 2px !important; color: black !important}
button {border-radius: 8px !important;}
.add_button {background-color: #4CAF50 !important;}
.remove_button {background-color: #f44336 !important;}
.mask_button_group {gap: 10px !important;}
.video {height: 300px !important;}
.image {height: 300px !important;}
.video .wrap.svelte-lcpz3o {display: flex !important; align-items: center !important; justify-content: center !important;}
.video .wrap.svelte-lcpz3o > :first-child {height: 100% !important;}
.margin_center {width: 50% !important; margin: auto !important;}
.jc_center {justify-content: center !important;}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as iface:
    click_state = gr.State([[],[]])

    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        }
    )

    gr.Markdown(title)
    gr.Markdown(description)
  
    with gr.Column():
        # input video
        gr.Markdown("## Step1: Upload video")
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):      
                video_input = gr.Video(elem_classes="video")
                extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary") 
            with gr.Column(scale=2):
                run_status = gr.HighlightedText(value=[("",""), ("Try to upload your video and click the Get video info button to get started!", "Normal")],
                                                color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"})
                video_info = gr.Textbox(label="Video Info")
                task_type = gr.Radio(
                        choices=["object_removal", "watermark_removal(draw mask)", "watermark_removal(upload mask)"],
                        value="object_removal",
                        label="Task type",
                        interactive=True,
                        visible=True,
                        min_width=100,
                        scale=1)
                
        
        # add masks
        step2_title = gr.Markdown("---\n## Step2: Add masks", visible=False)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                template_frame1 = gr.ImageEditor(type="pil",interactive=True, elem_id="template_frame", visible=False)
                temp_save = gr.ImageEditor(type="pil",interactive=True, elem_id="template_frame", visible=False)
                draw_or_upload_mask = gr.Image(type='pil', visible=False)
                image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
            with gr.Column(scale=1, elem_classes="jc_center"):
                run_status2 = gr.HighlightedText(value=[("",""), ("Try to upload your video and click the Get video info button to get started!", "Normal")],
                                                color_map={"Normal": "green", "Error": "red", "Clear clicks": "gray", "Add mask": "green", "Remove mask": "red"}, visible=False)
                with gr.Row():
                    with gr.Column(elem_classes="mask_button_group"):
                        clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False)
                        remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False, elem_classes="remove_button")
                        Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False, elem_classes="add_button")
                    point_prompt = gr.Radio(
                        choices=["Positive", "Negative"],
                        value="Positive",
                        label="Point prompt",
                        interactive=True,
                        visible=False,
                        min_width=100,
                        scale=1)
                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
            
        
        #add prompt
        step_prompt = gr.Markdown("---\n## Step3: Add your prompt", visible=True)
        with gr.Blocks():
            prompt = gr.Textbox(label="Prompt", value="photograph of a beautiful empty scene, highest quality settings")
            negative_prompt = gr.Textbox(label="Negative Prompt")

        # output video
        step4_title = gr.Markdown("---\n## Step4: Track masks and get the inpainting result", visible=False)
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                tracking_video_output = gr.Video(visible=False, elem_classes="video")
                tracking_video_predict_button = gr.Button(value="1. Tracking", visible=False, elem_classes="margin_center")
            with gr.Column(scale=2):
                inpaiting_video_output = gr.Video(visible=False, elem_classes="video")
                inpaint_video_predict_button = gr.Button(value="2. Inpainting", visible=False, elem_classes="margin_center")

    # first step: get the video information 
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, video_state, task_type
        ],
        outputs=[video_state, video_info, template_frame, template_frame1,
                 image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, Add_mask_button, template_frame, template_frame1,
                 tracking_video_predict_button, tracking_video_output, inpaiting_video_output, remove_mask_button, inpaint_video_predict_button, step2_title, step4_title,mask_dropdown, run_status, run_status2]
    )   

    # second step: select images from slider
    image_selection_slider.release(fn=select_template, 
                                   inputs=[image_selection_slider, video_state, interactive_state], 
                                   outputs=[template_frame, video_state, interactive_state, run_status, run_status2], api_name="select_image")
    track_pause_number_slider.release(fn=get_end_number, 
                                   inputs=[track_pause_number_slider, video_state, interactive_state], 
                                   outputs=[template_frame, interactive_state, run_status, run_status2], api_name="end_image")
    
    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status, run_status2]
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown, task_type, template_frame1],
        outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status, run_status2, draw_or_upload_mask, temp_save]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown, temp_save, template_frame1],
        outputs=[interactive_state, mask_dropdown, run_status, run_status2, template_frame1]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, interactive_state, mask_dropdown, task_type, draw_or_upload_mask],
        outputs=[tracking_video_output, video_state, interactive_state, run_status, run_status2]
    )

    # inpaint video from select image and mask
    inpaint_video_predict_button.click(
        fn=inpaint_video,
        inputs=[video_state, mask_dropdown, prompt, negative_prompt],
        outputs=[inpaiting_video_output, run_status, run_status2]
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status, run_status2]
    )
    
    # clear input
    video_input.change(
        fn=restart,
        inputs=[],
        outputs=[ 
            video_state,
            interactive_state,
            click_state,
            tracking_video_output, inpaiting_video_output,
            template_frame,
            tracking_video_predict_button, image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
            Add_mask_button, template_frame, tracking_video_predict_button, tracking_video_output, inpaiting_video_output, remove_mask_button,inpaint_video_predict_button, step2_title, step4_title, mask_dropdown, video_info, run_status, run_status2
        ],
        queue=False,
        show_progress=False)
    
    video_input.clear(
        fn=restart,
        inputs=[],
        outputs=[ 
            video_state,
            interactive_state,
            click_state,
            tracking_video_output, inpaiting_video_output,
            template_frame,
            tracking_video_predict_button, image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
            Add_mask_button, template_frame, tracking_video_predict_button, tracking_video_output, inpaiting_video_output, remove_mask_button,inpaint_video_predict_button, step2_title, step4_title, mask_dropdown, video_info, run_status, run_status2
        ],
        queue=False,
        show_progress=False)
    
    # points clear
    clear_button_click.click(
        fn = clear_click,
        inputs = [video_state, click_state,],
        outputs = [template_frame,click_state, run_status, run_status2],
    )

    # set example
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "./test_sample/", test_sample) for test_sample in ["test-sample0.mp4", "test-sample1.mp4", "test-sample2.mp4", "test-sample3.mp4", "test-sample4.mp4"]],
        inputs=[video_input],
    )
    gr.Markdown(article)

# iface.queue(concurrency_count=1)
iface.launch(debug=True)