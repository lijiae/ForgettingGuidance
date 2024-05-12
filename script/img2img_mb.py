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
        image_feature= pipe.image_processor.preprocess(image).to(device)

        caption=dataset.get_selfcaption(image_name) # 加载原有的
        prompt=dataset.get_prompt(image_name) # 加载一下新的
        decrease_content=nlp.locate_node(caption,prompt)
        print(caption)
        print(prompt)
        print(decrease_content)
        prompt_embeddings=pipe.control_encode_prompt(prompt,device,1,True)
        decrease_embeddings=pipe.control_encode_prompt(decrease_content,device,1,True)


        latents = pipe.prepare_latents(
            image_feature, latent_timestep, args.bz, 1, prompt_embeddings.dtype, device)
        
        do_classifier_free_guidance=7.5
        decrease_guidance=7.5
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                    # 预测一个噪声
                    noise_pred = pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeddings.to(device),
                        return_dict=False,
                    )[0]

                    latent_model_input_p = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input_p = pipe.scheduler.scale_model_input(latent_model_input, t)

                    # 
                    noise_pred_p = pipe.unet(
                        latent_model_input_p,
                        t,
                        encoder_hidden_states=decrease_embeddings.to(device),
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_uncond_po, noise_pred_text_po = noise_pred_p.chunk(2)
                        noise_pred = noise_pred_uncond -decrease_guidance*(noise_pred_text - noise_pred_uncond)+ do_classifier_free_guidance * (noise_pred_text_po - noise_pred_uncond_po)
                        # noise_pred = noise_pred_uncond + do_classifier_free_guidance * (noise_pred_text_po - noise_pred_uncond_po)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            image_new = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            result=pipe.image_processor.postprocess(image_new.detach())[0]
            result.save(os.path.join(args.save_path,image_name+".png"))
            
main()