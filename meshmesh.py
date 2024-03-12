import numpy as np
from PIL import Image
import torch


def replace_color(img, color_hex):
    print('color_hex', color_hex)
    if color_hex:
        color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    else:
        color = (255, 255, 255)

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
                "colorlist": ("STRING",{"default": "","multiline": False, "dynamicPrompts": False}), 
                "background": ("STRING", {"default": "#00ff00","multiline": False, "dynamicPrompts": False}),
            }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_to_image"

    def mask_to_image(self, mask, colorlist, background):
        colors = colorlist.split(",")
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
            mask_color = colors[i] if i < len(colors) else "#ffffff"
            colored_mask = replace_color(_mask, mask_color)
            composite_image.alpha_composite(colored_mask)

        composite_image.convert("RGB")
        image = np.array(composite_image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)
    

    
mode_list = ['HEX', 'DEC']

def hex_to_dec(inhex):
    rval = inhex[1:3]
    gval = inhex[3:5]
    bval = inhex[5:]
    rgbval = (int(rval, 16), int(gval, 16), int(bval, 16))
    return rgbval

class ColorPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("COLOR", {"default": "white"},),
                "mode": (mode_list,), 
            }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "picker"
    CATEGORY = "color"
    OUTPUT_NODE = True

    def picker(self, color, mode,):
        ret = color
        if mode == 'DEC':
            ret = hex_to_dec(color)
        return (ret,)
    


NODE_CLASS_MAPPINGS = {
    "MasksToColoredMasks": MasksToColoredMasks,
    "ColorPicker": ColorPicker
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "MasksToColoredMasks": "Masks to Colored Masks",
    "ColorPicker": "Color Picker"
}
