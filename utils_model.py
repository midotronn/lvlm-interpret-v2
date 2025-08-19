import logging
import base64
from io import BytesIO
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
from transformers import BitsAndBytesConfig

func_to_enable_grad = '_sample'
setattr(LlavaForConditionalGeneration, func_to_enable_grad, torch.enable_grad(getattr(LlavaForConditionalGeneration, func_to_enable_grad)))

try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

def get_processor_model(args):
    
    # Add OpenVLA/Embodied-CoT detection to choose the correct model and processor loading path.
    is_openvla = (
        ('openvla' in str(args.model_name_or_path).lower()) or 
        ('ecot' in str(args.model_name_or_path).lower()) or 
        ('embodied-cot' in str(args.model_name_or_path).lower())
    )

    # Load processor; enable trust_remote_code for OpenVLA-style models that require custom code.
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True if is_openvla else False
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    if args.load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif args.load_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        quant_config = None

    # Load model: use AutoModelForVision2Seq for OpenVLA, otherwise standard Llava model.
    if is_openvla:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16,
            quantization_config=quant_config, low_cpu_mem_usage=True, device_map=args.device_map,
            trust_remote_code=True
        )
        setattr(model, 'is_openvla', True)
        # Add container for cross-attention weights to enable Grad-CAM style relevancy
        setattr(model, 'enc_attn_weights_xattn', [])
        # Register hooks on decoder cross-attention modules to capture attention weights with gradients
        def xattn_forward_hook(module, inputs, output):
            attn = None
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
            if attn is not None:
                try:
                    attn.requires_grad_(True)
                    attn.retain_grad()
                    model.enc_attn_weights_xattn.append(attn)
                except Exception:
                    pass
            return output
        try:
            for name, m in model.named_modules():
                lname = name.lower()
                if ('cross' in lname or 'encoder_attn' in lname) and ('vision' not in lname):
                    try:
                        m.register_forward_hook(xattn_forward_hook)
                    except Exception:
                        pass
        except Exception:
            # Fallback: if model structure is unexpected, skip hooks gracefully
            pass
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16, 
            quantization_config=quant_config, low_cpu_mem_usage=True, device_map=args.device_map
        )
        if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'config'):
            model.vision_tower.config.output_attentions = True

    # Initialize attention containers; only register hooks for non-OpenVLA models.
    model.enc_attn_weights = []
    
    if not getattr(model, 'is_openvla', False):
        # Register language model self-attention hooks to capture encoder attentions.
        def forward_hook(module, inputs, output): 
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
            else:
                attn = None
            if attn is None:
                logger.error(
                    ("Attention weights were not returned for the encoder. "
                    "To enable, set output_attentions=True in the forward pass of the model. ")
                )
                return output
            attn.requires_grad_(True)
            attn.retain_grad()
            model.enc_attn_weights.append(attn)
            return output

        hooks_pre_encoder, hooks_encoder = [], []
        for layer in model.language_model.model.layers:
            hook_encoder_layer = layer.self_attn.register_forward_hook(forward_hook)
            hooks_pre_encoder.append(hook_encoder_layer)

        # Register vision tower hooks to capture vision attentions (for relevancy), when available.
        model.enc_attn_weights_vit = []

        def forward_hook_image_processor(module, inputs, output): 
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
            else:
                attn = None
            if attn is None:
                logger.warning(
                    ("Attention weights were not returned for the vision model. "
                     "Relevancy maps will not be calculated for the vision model. " 
                     "To enable, set output_attentions=True in the forward pass of vision_tower. ")
                )
                return output
            attn.requires_grad_(True)
            attn.retain_grad()
            model.enc_attn_weights_vit.append(attn)
            return output

        hooks_pre_encoder_vit = []
        if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'vision_model'):
            for layer in model.vision_tower.vision_model.encoder.layers:
                hook_encoder_layer_vit = layer.self_attn.register_forward_hook(forward_hook_image_processor)
                hooks_pre_encoder_vit.append(hook_encoder_layer_vit)
    else:
        setattr(model, 'enc_attn_weights_vit', [])
    
    return processor, model

def process_image(image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
    if image_process_mode == "Pad":
        def expand2square(pil_img, background_color=(122, 116, 104)):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
        image = expand2square(image)
    elif image_process_mode in ["Default", "Crop"]:
        pass
    elif image_process_mode == "Resize":
        image = image.resize((336, 336))
    else:
        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
    if max(image.size) > max_len:
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
    if return_pil:
        return image
    else:
        buffered = BytesIO()
        image.save(buffered, format=image_format)
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        return img_b64_str


def to_gradio_chatbot(state):
        ret = []
        for i, (role, msg) in enumerate(state.messages):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

def move_to_device(input, device='cpu'):

    if isinstance(input, torch.Tensor):
        return input.to(device).detach()
    elif isinstance(input, list):
        return [move_to_device(inp) for inp in input]
    elif isinstance(input, tuple):
        return tuple([move_to_device(inp) for inp in input])
    elif isinstance(input, dict):
        return dict( ((k, move_to_device(v)) for k,v in input.items()))
    else:
        raise ValueError(f"Unknown data type for {input.type}")

