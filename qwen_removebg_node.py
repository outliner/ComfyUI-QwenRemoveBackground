"""
阿里 Qwen-Image-Edit 去背景节点 for ComfyUI
使用 DashScope API 实现图片去背景
"""
import os
import base64
import json
import urllib.request
import urllib.error
import ssl
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

class QwenImageEditRemoveBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["qwen-image-edit-max", "qwen-image-edit-plus"], {"default": "qwen-image-edit-max"}),
                "prompt": ("STRING", {"default": "去除背景，保留主体，透明背景", "multiline": True}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def remove_background(self, image, model, prompt, api_key=""):
        # Get API key from input or environment
        if not api_key:
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        
        # Fallback to hardcoded key for testing
        if not api_key:
            # NOTE: Please set DASHSCOPE_API_KEY environment variable or provide API key in node input
        
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set. Please provide API key.")
        
        # Convert tensor to PIL
        if hasattr(image, 'numpy'):
            img_array = image.numpy()
        else:
            img_array = image
        
        # Handle batch - process first image
        if len(img_array.shape) == 4:
            img_array = img_array[0]
        
        # Convert CHW to HWC and uint8
        if img_array.shape[0] == 3:
            img_array = np.transpose(img_array, (1, 2, 0))
        
        img_array = (img_array * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img_array)
        
        # Save to temp file
        temp_path = "/tmp/temp_input.jpg"
        pil_img.save(temp_path, "JPEG")
        
        # Encode to base64
        with open(temp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Call DashScope API
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        data = {
            "model": model,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{img_b64}"},
                        {"text": prompt}
                    ]
                }]
            },
            "parameters": {
                "watermark": False
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        # Disable SSL verification for Ali OSS
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise Exception(f"API Error: {e.code} - {error_body}")
        
        # Get result image URL
        if result.get("output", {}).get("choices"):
            result_url = result["output"]["choices"][0]["message"]["content"][0].get("image")
        else:
            raise Exception(f"API Error: {result}")
        
        # Download result
        req2 = urllib.request.Request(result_url)
        with urllib.request.urlopen(req2, context=ctx, timeout=60) as response:
            result_data = response.read()
        
        # Save result to ComfyUI input folder and return as tensor
        import torchvision.transforms as transforms
        
        # Save result to temp location
        result_path = "/tmp/temp_output.png"
        with open(result_path, "wb") as f:
            f.write(result_data)
        
        # Load image using PIL
        result_img = Image.open(result_path)
        
        # Convert to RGB if RGBA
        if result_img.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', result_img.size, (255, 255, 255))
            # Composite RGBA image onto background
            background.paste(result_img, mask=result_img.split()[3])
            result_img = background
        elif result_img.mode != 'RGB':
            result_img = result_img.convert('RGB')
        
        # Convert to tensor in ComfyUI format [B, H, W, C]
        result_array = np.array(result_img).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_array).unsqueeze(0)  # [1, H, W, C]
        
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "QwenImageEditRemoveBackground": QwenImageEditRemoveBackground
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditRemoveBackground": "Qwen Image Edit Remove Background"
}
