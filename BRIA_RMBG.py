# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for background removal using various models,
# including RMBG-2.0, INSPYRENET, BEN, BEN2 and BIREFNET-HR. It leverages deep learning techniques
# to process images and generate masks for background removal.
#
# Models License Notice:
# - RMBG-2.0: Apache-2.0 License (https://huggingface.co/briaai/RMBG-2.0)
# - INSPYRENET: MIT License (https://github.com/plemeri/InSPyReNet)
# - BEN: Apache-2.0 License (https://huggingface.co/PramaLLC/BEN)
# - BEN2: Apache-2.0 License (https://huggingface.co/PramaLLC/BEN2)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-RMBG

import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import folder_paths
from PIL import ImageFilter
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import shutil
import sys
import importlib.util
from transformers import AutoModelForImageSegmentation
import cv2
import types
from .blurfusion_foreground_estimation import FB_blur_fusion_foreground_estimator_2

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

# Model configuration
AVAILABLE_MODELS = {
    "RMBG-2.0": {
        "type": "rmbg",
        "repo_id": "1038lab/RMBG-2.0",
        "files": {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py"
        },
        "cache_dir": "RMBG-2.0"
    }
}

# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def handle_model_error(message):
    print(f"[RMBG ERROR] {message}")
    raise RuntimeError(message)

class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")
    
    def get_cache_dir(self, model_name):
        cache_path = os.path.join(self.base_cache_dir, AVAILABLE_MODELS[model_name]["cache_dir"])
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    
    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"
        
        missing_files = []
        for filename in model_info["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, model_info["files"][filename])):
                missing_files.append(filename)
        
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
            
        return True, "Model cache verified"
    
    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")
            
            for filename in model_info["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=filename,
                    local_dir=cache_dir
                )
                    
            return True, "Model files downloaded successfully"
            
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"
    
    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model = None
        self.current_model_version = None

class RMBGModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()

            cache_dir = self.get_cache_dir(model_name)
            try:
                # Primary path: Modern transformers compatibility mode (optimized for newer versions)
                try:
                    from transformers import PreTrainedModel
                    import json

                    config_path = os.path.join(cache_dir, "config.json")
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    birefnet_path = os.path.join(cache_dir, "birefnet.py")
                    BiRefNetConfig_path = os.path.join(cache_dir, "BiRefNet_config.py")

                    # Load the BiRefNetConfig
                    config_spec = importlib.util.spec_from_file_location("BiRefNetConfig", BiRefNetConfig_path)
                    config_module = importlib.util.module_from_spec(config_spec)
                    sys.modules["BiRefNetConfig"] = config_module
                    config_spec.loader.exec_module(config_module)

                    # Fix and load birefnet module
                    with open(birefnet_path, 'r') as f:
                        birefnet_content = f.read()

                    birefnet_content = birefnet_content.replace(
                        "from .BiRefNet_config import BiRefNetConfig",
                        "from BiRefNetConfig import BiRefNetConfig"
                    )

                    module_name = f"custom_birefnet_model_{hash(birefnet_path)}"
                    module = types.ModuleType(module_name)
                    sys.modules[module_name] = module
                    exec(birefnet_content, module.__dict__)

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr != PreTrainedModel:
                            BiRefNetConfig = getattr(config_module, "BiRefNetConfig")
                            model_config = BiRefNetConfig()
                            self.model = attr(model_config)

                            weights_path = os.path.join(cache_dir, "model.safetensors")
                            try:
                                try:
                                    import safetensors.torch
                                    self.model.load_state_dict(safetensors.torch.load_file(weights_path))
                                except ImportError:
                                    from transformers.modeling_utils import load_state_dict
                                    state_dict = load_state_dict(weights_path)
                                    self.model.load_state_dict(state_dict)
                            except Exception as load_error:
                                pytorch_weights = os.path.join(cache_dir, "pytorch_model.bin")
                                if os.path.exists(pytorch_weights):
                                    self.model.load_state_dict(torch.load(pytorch_weights, map_location="cpu"))
                                else:
                                    raise RuntimeError(f"Failed to load weights: {str(load_error)}")
                            break

                    if self.model is None:
                        raise RuntimeError("Could not find suitable model class")

                except Exception as modern_e:
                    print(f"[RMBG INFO] Using standard transformers loading (fallback mode)...")
                    try:
                        self.model = AutoModelForImageSegmentation.from_pretrained(
                            cache_dir,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    except Exception as standard_e:
                        handle_model_error(f"Failed to load model with both modern and standard methods. Modern error: {str(modern_e)}. Standard error: {str(standard_e)}")

            except Exception as e:
                handle_model_error(f"Error loading model: {str(e)}")

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
            
    def process_image(self, images, model_name):
        try:
            self.load_model(model_name)

            # Prepare batch processing
            transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]

            original_sizes = [tensor2pil(img).size for img in images]

            input_tensors = [transform_image(tensor2pil(img)).unsqueeze(0) for img in images]
            input_batch = torch.cat(input_tensors, dim=0).to(device)

            with torch.no_grad():
                outputs = self.model(input_batch)
                
                if isinstance(outputs, list) and len(outputs) > 0:
                    results = outputs[-1].sigmoid().cpu()
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    results = outputs['logits'].sigmoid().cpu()
                elif isinstance(outputs, torch.Tensor):
                    results = outputs.sigmoid().cpu()
                else:
                    try:
                        if hasattr(outputs, 'last_hidden_state'):
                            results = outputs.last_hidden_state.sigmoid().cpu()
                        else:
                            for k, v in outputs.items():
                                if isinstance(v, torch.Tensor):
                                    results = v.sigmoid().cpu()
                                    break
                    except:
                        handle_model_error("Unable to recognize model output format")
                
                masks = []
                
                for i, (result, (orig_w, orig_h)) in enumerate(zip(results, original_sizes)):
                    result = result.squeeze()
                    result = torch.clamp(result, 0, 1)

                    result = F.interpolate(result.unsqueeze(0).unsqueeze(0),
                                         size=(orig_h, orig_w),
                                         mode='bilinear').squeeze()
                    
                    masks.append(tensor2pil(result))

                return masks

        except Exception as e:
            handle_model_error(f"Error in batch processing: {str(e)}")


class RMBG:
    def __init__(self):
        self.models = {
            "RMBG-2.0": RMBGModel()
        }
    
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "image": "Input image to be processed for background removal.",
            "model": "Select the background removal model to use (RMBG-2.0, INSPYRENET, BEN)."
        }
        
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "model": (list(AVAILABLE_MODELS.keys()), {"tooltip": tooltips["model"]}),
            },
            "optional": {
                "edge_optimization": ("BOOLEAN", {"default": True}),
                "blur_ksize": ("INT", {"default": 90, "min": 0, "max": 90, "step": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process_image"
    CATEGORY = "🧪BRIA_RMBG-2.0"

    def process_image(self, image, model, edge_optimization=True, blur_ksize=90):
        try:
            processed_images = []
            processed_masks = []
            
            model_instance = self.models[model]
            
            cache_status, message = model_instance.check_model_cache(model)
            if not cache_status:
                print(f"Cache check: {message}")
                print("Downloading required model files...")
                download_status, download_message = model_instance.download_model(model)
                if not download_status:
                    handle_model_error(download_message)
                print("Model files downloaded successfully")
            
            def _process_pair(img, mask):
                if isinstance(mask, list):
                    masks = [m.convert("L") for m in mask if isinstance(m, Image.Image)]
                    mask_local = masks[0] if masks else None
                elif isinstance(mask, Image.Image):
                    mask_local = mask.convert("L")
                else:
                    mask_local = mask
                
                mask_tensor_local = pil2tensor(mask_local)
                mask_tensor_local = torch.clamp(mask_tensor_local, 0, 1)
                mask_img_local = tensor2pil(mask_tensor_local)
                
                orig_image_local = tensor2pil(img)
                orig_rgba_local = orig_image_local.convert("RGBA")
                r, g, b, _ = orig_rgba_local.split()
                

                if edge_optimization:
                    foreground_local = refine_foreground(orig_image_local, mask_img_local, r=blur_ksize)
                else:
                    foreground_local = Image.merge('RGBA', (r, g, b, mask_img_local))

                processed_images.append(pil2tensor(foreground_local))
                processed_masks.append(pil2tensor(mask_img_local))
            
            
            images_list = [img for img in image]
            chunk_size = 4
            for start in range(0, len(images_list), chunk_size):
                batch_imgs = images_list[start:start + chunk_size]
                masks = model_instance.process_image(batch_imgs, model)
                if isinstance(masks, Image.Image):
                    masks = [masks]
                for img_item, mask_item in zip(batch_imgs, masks):
                    _process_pair(img_item, mask_item)
           
            mask_images = []
            for mask_tensor in processed_masks:
                mask_image = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                mask_images.append(mask_image)
            
            mask_image_output = torch.cat(mask_images, dim=0)
            return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0), mask_image_output)
            
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}")
            empty_mask = torch.zeros((image.shape[0], image.shape[2], image.shape[3]))
            empty_mask_image = empty_mask.reshape((-1, 1, empty_mask.shape[-2], empty_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            return (image, empty_mask, empty_mask_image)

def refine_foreground(image: Image.Image, mask: Image.Image, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    alpha_array = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_2(
        image_array, alpha_array, r=r
    )
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    image_masked.putalpha(mask.resize(image.size))
    return image_masked

NODE_CLASS_MAPPINGS = {
    "RMBG": RMBG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RMBG": "Remove Background(RMBG)"
} 
