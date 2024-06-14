import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from lavis.datasets.datasets.dataset_3d import ModelNet
from lavis.common.sample import BatchCollator
from torch.utils.data import DataLoader

def test_mm_pc():
    '''
    ViT-Lens Integration:
        point cloud -> InstructBLIP
        
    Describe a 3D point cloud object.
    '''
    model, _, _ = load_model_and_preprocess(
        name="pc_blip2_vicuna_instruct", 
        model_type="vicuna7b", 
        is_eval=True, 
        device="cuda:1"
    )
    
    prompts = [
        "What is this object?",
        "What is the function of this object?",
        "Describe this object in details.",
        "Describe this object briefly.",
    ]
    
    result = dict()
    # dataset from sfr-ulip-code-release-research (see https://github.com/salesforce/ULIP)
    dataset = ModelNet(root="/PATH_TO/modelnet40_normal_resampled", split="test")
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, drop_last=False, collate_fn=BatchCollator(dataset_type="test"))
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.to(model.device)
        for prompt in prompts:
            output = model.generate({"image": batch["pc"], "prompt": prompt}, modality_type="pc")
            for sample_idx, inst_name, response in zip(batch["sample_idx"], batch["instance_name"], output):
                key = f"{sample_idx}_{inst_name}"
                if not key in result:
                    result[key] = []
                result[key].append({prompt: response})

    with open("./outputs/modelnet_pc_instructblip.json", "w") as f:
        json.dump(result, f, indent=2)


def test_fuse_mm2(cuda_idx=1):
    '''
    ViT-Lens Integration:
        point cloud + image -> InstructBLIP
        
    Describe / chat with multimodal inputs (point cloud + image).
    '''
    prompts = [
        "Describe what you see.",
        "What is unusual about this image?",
        "Create a short story around the given visual contents.",
        "Imagine you are a narrator looking at this, write a story that captures the essence of what you see.",
        "Craft a narrative based on the visual elements in the picture. Let your imagination guide the story.",
        "Incorporate the details you see into a creative story. Feel free to add characters, emotions, and dialogue.",
        "Tell a story based on what you see.",
        "Imagine the events leading up to this moment and those that follow. Create a story that links them together.",
    ]

    cases = [
        ("space", "person_0102"),
        ("sea", "person_0102"),
        ("beach", "person_0102"),
        ("astronaut", "piano_0286"),
        ("space", "piano_0286"),
        ("astronaut", "car_0260"),
        ("astronaut", "guitar_0243"),
        ("space", "car_0260"),
        ("sea", "car_0260"),
        ("road", "car_0260"),
        ("space", "guitar_0243"),
        ("sea", "monitor_0503"),
        ("dog", "monitor_0503"),
        ("sea", "monitor_0503"),
    ]
    # dataset from sfr-ulip-code-release-research (see https://github.com/salesforce/ULIP)
    dataset = ModelNet(root="/PATH_TO/modelnet40_normal_resampled", split="test")
    model, vis_processors, _ = load_model_and_preprocess(
        name="pc_blip2_vicuna_instruct", 
        model_type="vicuna7b", 
        is_eval=True, 
        device=f"cuda:{cuda_idx}"
    )
    
    result_list = []
    for im, pc_inst in cases:
        img = Image.open(f"./assets/{im}.jpg").convert("RGB")
        item = None
        for i in range(len(dataset)):
            item = dataset[i]
            if item["instance_name"] == pc_inst:
                break

        img = vis_processors["eval"](img).unsqueeze(0).to(model.device)
        pc = item["pc"].unsqueeze(0).to(model.device)
        
        for prompt in prompts:
            for fusion_type in ["vit", "qformer", "llm"]:
                output = model.generate_mm(
                    samples = {
                        "modality_input": {"image": img, "pc": pc},
                        "prompt": prompt,
                    },
                    fuse_before = fusion_type
                )
                result_list.append(
                    {"meta_info": f"{im},{pc_inst}", "prompt": prompt, "fusion":fusion_type, "output": output}
                )
    with open("./outputs/modelnet_pc_instructblip_mm2-released_test.json", "w") as f:
        json.dump(result_list, f, indent=2)
        

if __name__ == "__main__":
    test_fuse_mm2()
            