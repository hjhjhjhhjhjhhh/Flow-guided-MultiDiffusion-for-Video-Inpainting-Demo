import torch
import cv2
import os
import glob
from diffusers.utils import load_image, make_image_grid
from inpainter.new_pipeline import VideoDiffusionInpaintPipeline
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import DDIMScheduler, StableDiffusionPipeline
import argparse
from moviepy.editor import VideoFileClip
from inpainter.compute_flow import compute_flow
import utils
import utils.flow_util

def load_images(folder_path, type="jpg", all=True):
    # Retrieve image file names
    images_path = sorted(glob.glob(f"{folder_path}/*.{type}"))
    images = []
    if all:
        for img_path in images_path:
            img = load_image(img_path)
            images.append(img)
    else:
        for i in range(20):
            img = load_image(images_path[i])
            images.append(img)
    return images

def save_images(images, folder_path, type="jpg"):
    # Save images
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, img in enumerate(images):
        if i < 10:
            save_path = folder_path + '/0000' + str(i) + '.' + type
        elif i < 100:
            save_path = folder_path + '/000' + str(i) + '.' + type
        else:
            save_path = folder_path + '/00' + str(i) + '.' + type
        img.save(save_path) 

def save_video(save_dir, frame_rate=10, extension='jpg'):
    # Determine the width and height from the first image
    output = save_dir + '/output_video.mp4'
    # Retrieve image file names
    images = sorted(glob.glob(f"{save_dir}/*.{extension}"))

    # Determine the width and height from the first image
    image_path = images[0]
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video = cv2.VideoWriter(output, fourcc, frame_rate, (width, height))

    # Add images to video
    for image in images:
        video.write(cv2.imread(image))
    video.release()

def save_gif(save_dir, frame_rate=10):
    # Load the video clip
    clip = VideoFileClip(save_dir + '/output_video.mp4')

    # Set the frame rate
    clip = clip.set_fps(frame_rate)

    # Convert the video to GIF
    clip.write_gif(save_dir + '/output_video.gif')


def imgs_preprocessing(images, resolution=512, type_="resize"):
    # Preprocessing images and masks
    init_imgs = []
    if type_ == "resize":
        for img in images:
            init_imgs.append(img.resize((resolution, resolution), Image.BILINEAR))
    elif type_ == "cropping":
        for img in images:
            original_width, original_height = img.size
            # Desired dimensions
            target_width = 480
            target_height = 480
            # Calculate coordinates to crop the center
            left = (original_width - target_width) / 2
            top = (original_height - target_height) / 2
            right = (original_width + target_width) / 2
            bottom = (original_height + target_height) / 2
            init_imgs.append(img.crop((left, top, right, bottom)).resize((512,512)))
    return init_imgs

def masks_preprocessing(masks, resolution=512, type="resize", ):
    # Preprocessing images and masks
    mask_imgs = []
    if type == "resize":     
        for mask in masks:
            mask_imgs.append(mask.resize((resolution, resolution), Image.BILINEAR))
    elif type == "cropping":
        for mask in masks:
            original_width, original_height = mask.size
            # Desired dimensions
            target_width = 480
            target_height = 480
            # Calculate coordinates to crop the center
            left = (original_width - target_width) / 2
            top = (original_height - target_height) / 2
            right = (original_width + target_width) / 2
            bottom = (original_height + target_height) / 2
            mask_imgs.append(mask.crop((left, top, right, bottom)).resize((512,512)))
    return mask_imgs

def imgs_postprocessing(images, original_imgs, type_="resize"):
    # Postprocessing images
    post_imgs = []
    if type_ == "resize":
        for img, original_img in zip(images, original_imgs):
            post_imgs.append(img.resize(original_img.size, Image.BILINEAR))
    elif type_ == "cropping":
        post_imgs = images
    return post_imgs

class StableDiffusionInpainting:
    def __init__(self, model_path, strength=1, steps=20):
        self.model_path = model_path
        self.strength = strength
        self.steps = steps
        
    def run(self, frames, inpaint_masks, prompt, negative_prompt):
        # Load images
        
        original_images = []
        original_masks = []
        for i in range(len(frames)):
            original_images.append(Image.fromarray(frames[i]))
            original_masks.append(Image.fromarray(inpaint_masks[i][:, :, 0]))

        # Preprocessing images and masks
        imgs = imgs_preprocessing(original_images, 512, "resize")
        masks = masks_preprocessing(original_masks, 512, "resize")
        
        """
        The part to add addtional processing for the images and masks
        ie. feature space optical flow warping
        !!! make sure the size is 512x512
        """

        updated_frames, updated_masks, pred_flows_bi, updated_frames_tensor, updated_masks_tensor = compute_flow(imgs, masks)
        # utils.flow_util.visualize_flow(np.array(updated_frames[0]), pred_flows_bi[0], self.output_path, 'forward')
        # utils.flow_util.visualize_flow(np.array(updated_frames[0]), pred_flows_bi[1], self.output_path, 'backward')
        # Save images
        #save_images(updated_frames, self.output_path, "jpg")
        # Save masks
        #save_images(updated_masks, self.output_path + '/mask', "png")
        #return None 
        # Load model
        imgs = updated_frames
        masks = updated_masks
        pipeline = VideoDiffusionInpaintPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16, variant="fp16"
        )
        pipeline = pipeline.to("cuda")
        pipeline.enable_model_cpu_offload()

        # Args
        generator = torch.Generator("cuda").manual_seed(92)
        # prompt = "photograph of a beautiful empty scene, highest quality settings"

        # Inpainting
        images = pipeline(
            prompt=prompt,
            images=imgs,
            mask_images=masks,
            generator=generator,
            num_inference_steps=self.steps,
            negative_prompt=negative_prompt,
            flows=pred_flows_bi,
            do_multi_diffusion=True,
            strength=self.strength,
            updated_frames_tensor=updated_frames_tensor,
            updated_masks_tensor=updated_masks_tensor,
        )

        # Postprocessing images
        temp_images = []
        for img in images: 
            temp_images.append(img[0])
        images = temp_images
        images = imgs_postprocessing(images, original_images, "resize")
        
        for i in range(len(images)):
            images[i] = np.asarray(images[i])
        
        torch.cuda.empty_cache()
        return images


class BlendedLatentDiffusionInpainting:
    def __init__(self, model_path, image_path, mask_path, output_path, strength=1, steps=50):
        self.model_path = model_path
        self.image_path = image_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.strength = strength
        self.steps = steps

    def load_models(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.float16
        )
        self.vae = pipe.vae.to("cuda")
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to("cuda")
        self.unet = pipe.unet.to("cuda")
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    def run(self):
        # Load images
        original_imgs = load_images(self.image_path, "jpg")
        original_masks = load_images(self.mask_path, "png")
        original_size = original_imgs[0].size

        # Preprocessing images and masks
        imgs = imgs_preprocessing(original_imgs, 512, "resize")
        masks = masks_preprocessing(original_masks, 512, "resize")

        source_latents_list = []
        latents_list = []
        for img in imgs:
            img = np.array(img)[:, :, :3]
            source_latents = self._image2latent(img)
            source_latents_list.append(source_latents)
            latents = torch.randn(
                (1, self.unet.in_channels, 512 // 8, 512 // 8),
                generator=torch.manual_seed(42),
            )
            latents = latents.to("cuda").half()
            latents_list.append(latents)

        latent_mask_list = []
        for mask in masks:
            latent_mask = mask.convert("L").resize((64, 64), Image.BILINEAR)
            latent_mask = np.array(latent_mask) / 255
            latent_mask[latent_mask < 0.5] = 0
            latent_mask[latent_mask >= 0.5] = 1
            latent_mask = latent_mask[np.newaxis, np.newaxis, ...]
            latent_mask = torch.from_numpy(latent_mask).half().to("cuda")
            latent_mask_list.append(latent_mask)

        text_input = self.tokenizer(
            "photograph of a beautiful empty scene, highest quality settings",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * 1,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.scheduler.set_timesteps(self.steps)


        blending_percentage = 0.0
        numframe = len(imgs)
        for t in tqdm(self.scheduler.timesteps[
            int(len(self.scheduler.timesteps) * blending_percentage) :
        ]):
            for i in tqdm(range(numframe)):

                # expand the latents
                latent_model_input = torch.cat([latents_list[i]] * 2)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep=t
                )

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample
                
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (
                    noise_pred_text - noise_pred_uncond
                )
                # compute the previous noisy sample x_t -> x_t-1
                latents_list[i] = self.scheduler.step(noise_pred, t, latents_list[i]).prev_sample

                # blending
                noise_source_latents = self.scheduler.add_noise(
                    source_latents_list[i], torch.randn_like(latents_list[i]), t
                )
                latents_list[i] = latents_list[i] * latent_mask_list[i] + noise_source_latents * (1 - latent_mask_list[i])

        images = []
        for i in range(numframe):
            latents_list[i] = 1 / 0.18215 * latents_list[i]
            with torch.no_grad():
                image = self.vae.decode(latents_list[i]).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                image = (image * 225).round().astype("uint8")
                images.append(Image.fromarray(image[0]))
        
        images = imgs_postprocessing(images, original_imgs, "resize")
        save_images(images, self.output_path, "jpg")

        # Save video
        frame_rate = 10
        save_video(self.output_path, frame_rate)

        # Save gif
        save_gif(self.output_path, frame_rate)


    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents  



if __name__ == '__main__':
    """
    Stable Diffusion for Inpaint:
        1. stable-diffusion-inpainting
        2. Blended Latent Diffusion
    
    Type of image transform:
        1. resize
        2. cropping
    
    """
    run_all = False
    model_type = "stable-diffusion-inpainting"
    #model_type = "blended-latent-diffusion"
    image_path = "../../Downloads/DAVIS-data/DAVIS/JPEGImages/480p/"
    mask_path = "../../Downloads/DAVIS-data/DAVIS/Annotations/480p/"
    #mask_path = "data/test_masks/"
    #image_path = 'data/completed_flow/'
    #mask_path = 'data/completed_flow/'
    output_path = "data/output/StableInpaint/multidiffuse_allframe_200step/"
    #output_path = "data/output/BlendedInpaint/temp_fixcode/"
    if model_type == "stable-diffusion-inpainting":

        if run_all:
            count = 0
            # List all items in the directory and filter out files, keeping only directories
            folder_names = [item for item in os.listdir(image_path)
                            if os.path.isdir(os.path.join(image_path, item))]
            for folder_name in folder_names:
                count += 1
                print("Count: ", count)
                print("=====================================")
                print("Processing: ", folder_name)
                print("=====================================")
                run_inpaint = StableDiffusionInpainting( 
                    "runwayml/stable-diffusion-inpainting",
                    image_path + folder_name,
                    mask_path + folder_name,
                    output_path + folder_name,
                    
                )
                run_inpaint.run()

        else:
            #folder_names = "bear"
            folder_names = "bear"
            run_inpaint = StableDiffusionInpainting( 
                "runwayml/stable-diffusion-inpainting",
                "../../Downloads/DAVIS-data/DAVIS/JPEGImages/480p/" + folder_names,
                "../../Downloads/DAVIS-data/DAVIS/Annotations/480p/" + folder_names,
                "data/output1/" + folder_names,
                steps=20
            )
            run_inpaint.run()
    elif model_type == "blended-latent-diffusion":
        if run_all:
            # List all items in the directory and filter out files, keeping only directories
            folder_names = [item for item in os.listdir(image_path)
                            if os.path.isdir(os.path.join(image_path, item))]
            print(folder_names)
            for folder_name in folder_names:
                print("=====================================")
                print("Processing: ", folder_name)
                print("=====================================")
                run_inpaint = BlendedLatentDiffusionInpainting( 
                    "stabilityai/stable-diffusion-2-1-base",
                    image_path + folder_name,
                    mask_path + folder_name,
                    output_path + folder_name,
                    strength=1,
                    steps=50
                )
                run_inpaint.load_models()
                run_inpaint.run()
        else:
            folder_names = ["tennis"]
            run_inpaint = BlendedLatentDiffusionInpainting( 
                "runwayml/blended-latent-diffusion",
                "data/img/480p/tennis", 
                "data/mask_img/480p/tennis",
                "data/output/BlendedInpaint/480p/tennis"
            )
            run_inpaint.load_models()
            run_inpaint.run()