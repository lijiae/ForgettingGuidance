import sys
sys.path.append('/root/autodl-tmp/lijia')
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from diffusers import DDIMScheduler
import argparse
import os
from model_pipeline.locate_word import TreeLocate
from utils.MagicBrush import MaginBrush
from tqdm import tqdm

def make_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,default="cuda")
    parser.add_argument('--bz', type=int,default=1)
    parser.add_argument('--width', type=int,default=512)
    parser.add_argument('--height', type=int,default=512)
    parser.add_argument('--seed', type=int,default=1999)
    parser.add_argument('--save_path',type=str,default="/root/autodl-tmp/lijia/output/magicbrush")
    parser.add_argument('--image_dir',type=str,default="/root/autodl-tmp/lijia/data/test/images")
    parser.add_argument('--config_path',type=str,default="/root/autodl-tmp/lijia/data/test/global_descriptions.json")

    parser.add_argument('--num_inference_steps', type=int,default=50)
    parser.add_argument('--model_name',type=str,default="runwayml/stable-diffusion-v1-5")
    args = parser.parse_args()
    return args

def load_model(ckpt_name):
    pipe=StableDiffusionImg2ImgPipeline.from_pretrained(ckpt_name)
    print("load model!")
    return pipe

def read_image(image_path,pipe):
    image=load_image(image_path)
    return image

def main():
    args=make_argparse()
    device=args.device
    pipe=load_model(args.model_name).to(device)
    torch.manual_seed(args.seed)
    
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps, num_inference_steps = pipe.get_timesteps(args.num_inference_steps, 0.8, device)
    latent_timestep = timesteps[:1].repeat(args.bz * 1)

    nlp=TreeLocate()
    dataset=MaginBrush(args.config_path,args.image_dir)
    print("load datatset!")
    
    image_list=dataset.image_names
    for image_name in tqdm(image_list):
        image=load_image(os.path.join(args.image_dir,image_name,image_name+"-input.png"))
        # image_feature= pipe.image_processor.preprocess(image).to(device)

        prompt=dataset.get_prompt(image_name) # 加载一下新的
        pipe(prompt=prompt, image=image,num_inference_steps=args.num_inference_steps,num_images_per_prompt=1).images[0].save(os.path.join(args.save_path,image_name+".png"))
            
main()