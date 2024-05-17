from open_flamingo import create_model_and_transforms
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests
import torch
import json
import os


def load_model(device="cuda"):
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
    )

    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model=model.to(device)

    return model,image_processor, tokenizer

def get_caption_from_image(model,image_processor,tokenizer,image_path,device="cuda"):
    """
    Step 1: Load images
    """
    query_image = Image.open(image_path)

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = 1, num_media = 3, num_frames = 1,
    channels = 3, height = 224, width = 224.
    """
    vision_x = [image_processor(query_image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = tokenizer(
        ["<image>"],
        return_tensors="pt",
    )

    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x.to(device),
        lang_x=lang_x["input_ids"].to(device),
        attention_mask=lang_x["attention_mask"].to(device),
        max_new_tokens=20,
        num_beams=1,
    )

    # print("Generated text: ", tokenizer.decode(generated_text[0]))
    return tokenizer.decode(generated_text[0])

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():
    image_dir="/root/autodl-tmp/data/imagic-editing.github.io/tedbench/originals"
    path="/root/autodl-tmp/data/imagic-editing.github.io/tedbench/input_list.json"
    all_images=read_json_file(path)

    # load model
    model,image_processor, tokenizer=load_model()

    result=[]

    for d in all_images:
        name=d["img_name"]
        prompt=d["target_text"]
        image_path=os.path.join(image_dir,name)

        caption=get_caption_from_image(model,image_processor, tokenizer,image_path)
        print(caption)

        result.append({
            "img_name":name,
            "target_text":prompt,
            "caption":caption
        })

        # 设置保存的文件路径
    file_path = '/root/autodl-tmp/data/imagic-editing.github.io/tedbench/test_list.json'

    # 将列表保存为 JSON 文件
    with open(file_path, 'w') as f:
        json.dump(result, f)
        
    print("JSON 文件保存成功！")



main()