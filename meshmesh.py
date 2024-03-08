import numpy as np
from PIL import Image
import torch


def replace_color(img, color_hex):

    color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    red, green, blue, alpha = img.T

    # Replace non-black with color and set black to alpha
    black_pixels = (red == 0) & (green == 0) & (blue == 0)

    alpha[black_pixels] = 0
    img[..., :-1][~black_pixels.T] = color

    return Image.fromarray(img)


class MasksToColoredMasks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "colorlist": ("STR",),
                "background": ("COLOR", {"default": "#00ff00"}),
            }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_to_image"

    def mask_to_image(self, mask, colorlist, background):
        colors = colorlist["string"].split(",")
        result = mask.reshape(
            # (Masks, Add Channels Dim, Height, Width)
            (-1, 1, mask.shape[-2], mask.shape[-1])
            ).movedim(1, -1  # Reorder to (Masks, Height, Width, Channels)
            ).expand(-1, -1, -1, 4)  # Set Channels to 4 (RGBA)

        # create a new empty image in pil
        res = (mask.shape[-1], mask.shape[-2])
        composite_image = Image.new("RGBA", res, color=background)

        for i, _mask in enumerate(result):
            _mask = (_mask.numpy() * 255).astype(np.uint8)
            mask_color = colors[i] if i < len(colors) else (255, 255, 255)
            colored_mask = replace_color(_mask, mask_color)
            composite_image.alpha_composite(colored_mask)

        composite_image.convert("RGB")
        image = np.array(composite_image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)
    

class ColorPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background": ("COLOR", {"default": "#ff0000"}),
            }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("COLOR",)
    FUNCTION = "run"

    def run(self, background):
        return (background,)
    
    


NODE_CLASS_MAPPINGS = {
    "MasksToColoredMasks": MasksToColoredMasks,
    "ColorPicker": ColorPicker
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MasksToColoredMasks": "Masks to Colored Masks",
    "ColorPicker": "Color Picker"
}
