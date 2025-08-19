import os
import tempfile
import logging

import torch

from PIL import Image
import numpy as np
import gradio as gr
import spaces

from torchvision.transforms.functional import to_pil_image

from utils_model import get_processor_model, move_to_device, to_gradio_chatbot, process_image

from utils_attn import (
    handle_attentions_i2t, plot_attention_analysis, handle_relevancy, handle_text_relevancy, reset_tokens,
    plot_text_to_image_analysis, handle_box_reset, boxes_click_handler, attn_update_slider
)

from utils_relevancy import construct_relevancy_map, construct_relevancy_map_openvla

from utils_causal_discovery import (
    handle_causality, handle_causal_head, causality_update_dropdown
)

logger = logging.getLogger(__name__)

N_LAYERS = 32 
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROLE0 = "USER"
ROLE1 = "ASSISTANT"

processor = None
model = None

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

title_markdown = ("""
# LVLM-Interpret: An Interpretability Tool for Large Vision-Language Models
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
""")

block_css = """

#image_canvas canvas {
    max-width: 400px !important;
    max-height: 400px !important;
}

#buttons button {
    min-width: min(120px,100%);
}

"""

def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = gr.State()
    state.messages = []
    return (state, [], "", None, None, None, None)

def add_text(state, text, image, image_process_mode):
    global processor
    
    if True: # state is None:
        state = gr.State()
        state.messages = []
        
    if isinstance(image, dict):
        image = image['composite']
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert('RGB')

        # ImageEditor does not return None image
        if (np.array(image)==255).all():
            image =None

    text = text[:1536]  # Hard cut-off
    logger.info(text)

    prompt_len = 0
    if processor.tokenizer.chat_template is not None:
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<image>\n" + text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len += len(prompt)
    else:
        prompt = system_prompt
        prompt_len += len(prompt)
        if image is not None:
            msg = f"\n{ROLE0}: <image>\n{text}\n{ROLE1}:" 
        else:
            msg = f"\n{ROLE0}: {text}\n{ROLE1}: "
        prompt += msg
        prompt_len += len(msg)

    state.messages.append([ROLE0,  (text, image, image_process_mode)])
    state.messages.append([ROLE1, None])

    state.prompt_len = prompt_len
    state.prompt = prompt
    state.image = process_image(image, image_process_mode, return_pil=True)

    return (state, to_gradio_chatbot(state), "", None)


@spaces.GPU
def lvlm_bot(state, temperature, top_p, max_new_tokens):   
    prompt = state.prompt
    prompt_len = state.prompt_len
    image = state.image
    
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Support both standard LVLM (with image token index) and OpenVLA-style models without explicit image token index.
    is_openvla = getattr(model, 'is_openvla', False)
    if not is_openvla and hasattr(model, 'config') and hasattr(model.config, 'image_token_index'):
        img_idx = torch.where(input_ids==model.config.image_token_index)[1][0].item()
    else:
        img_idx = None
    do_sample = True if temperature > 0.001 else False
    # Reset attention holders for this generation run.
    model.enc_attn_weights = []
    model.enc_attn_weights_vit = []

    # Choose proper EOS depending on backend (e.g., Gemma chat templates).
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'config') and getattr(model.language_model.config, 'model_type', None) == "gemma":
        eos_token_id = processor.tokenizer('<end_of_turn>', add_special_tokens=False).input_ids[0]
    else:
        eos_token_id = getattr(processor.tokenizer, 'eos_token_id', None)

    outputs = model.generate(
            **inputs, 
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=eos_token_id
        )

    # Build tokenized prompt for relevancy analysis when available.
    input_text_tokenized = []
    if not is_openvla and img_idx is not None:
        input_ids_list = input_ids.reshape(-1).tolist()
        input_ids_list[img_idx] = 0
        input_text = processor.tokenizer.decode(input_ids_list)
        if input_text.startswith("<s> "):
            input_text = '<s>' + input_text[4:]
        input_text_tokenized = processor.tokenizer.tokenize(input_text)
        if img_idx < len(input_text_tokenized):
            input_text_tokenized[img_idx] = "average_image"
    
    output_ids = outputs.sequences.reshape(-1)[input_ids.shape[-1]:].tolist()  

    generated_text = processor.tokenizer.decode(output_ids)
    output_ids_decoded = [processor.tokenizer.decode(oid).strip() for oid in output_ids]
    generated_text_tokenized = processor.tokenizer.tokenize(generated_text)

    logger.info(f"Generated response: {generated_text}")
    logger.debug(f"output_ids_decoded: {output_ids_decoded}")
    logger.debug(f"generated_text_tokenized: {generated_text_tokenized}")

    state.messages[-1][-1] = generated_text[:-len('</s>')] if generated_text.endswith('</s>') else generated_text

    tempdir = os.getenv('TMPDIR', '/tmp/')
    tempfilename = tempfile.NamedTemporaryFile(dir=tempdir)
    tempfilename.close()

    # Save input_ids and any attentions available from different backends (decoder vs cross-attn).
    fn_input_ids = f'{tempfilename.name}_input_ids.pt'
    torch.save(move_to_device(input_ids, device='cpu'), fn_input_ids)
    attn_obj = getattr(outputs, 'attentions', None)
    if attn_obj is None:
        attn_obj = getattr(outputs, 'decoder_attentions', None)
    if attn_obj is not None:
        fn_attention = f'{tempfilename.name}_attn.pt'
        torch.save(move_to_device(attn_obj, device='cpu'), fn_attention)
        logger.info(f"Saved attention to {fn_attention}")
    cross_attn_obj = getattr(outputs, 'cross_attentions', None)
    if cross_attn_obj is not None:
        fn_xattention = f'{tempfilename.name}_xattn.pt'
        torch.save(move_to_device(cross_attn_obj, device='cpu'), fn_xattention)
        logger.info(f"Saved cross-attention to {fn_xattention}")

    # Compute relevancy only for standard LVLM models; OpenVLA path may not support it directly.
    if not is_openvla:
        word_rel_map = construct_relevancy_map(
            tokenizer=processor.tokenizer, 
            model=model,
            input_ids=inputs.input_ids,
            tokens=generated_text_tokenized, 
            outputs=outputs, 
            output_ids=output_ids,
            img_idx=img_idx
        )
        fn_relevancy = f'{tempfilename.name}_relevancy.pt'
        torch.save(move_to_device(word_rel_map, device='cpu'), fn_relevancy)
        logger.info(f"Saved relevancy map to {fn_relevancy}")
    else:
        # For OpenVLA, compute relevancy from cross-attention over encoder keys
        try:
            state.image_idx = 0 if img_idx is None else img_idx
            rel_map_openvla = construct_relevancy_map_openvla(
                tokenizer=processor.tokenizer,
                model=model,
                outputs=outputs,
                tokens=generated_text_tokenized,
                enc_grid_rows=getattr(state, 'enc_grid_rows', 24) or 24,
                enc_grid_cols=getattr(state, 'enc_grid_cols', 24) or 24,
                apply_normalization=True,
                enc_kv_ignore_first=getattr(state, 'enc_kv_ignore_first', 0)
            )
            fn_relevancy = f'{tempfilename.name}_relevancy.pt'
            torch.save(move_to_device(rel_map_openvla, device='cpu'), fn_relevancy)
            logger.info(f"Saved OpenVLA relevancy map to {fn_relevancy}")
        except Exception as e:
            logger.warning(f"OpenVLA relevancy map failed: {e}")
    model.enc_attn_weights = []
    if hasattr(model, 'enc_attn_weights_vit'):
        model.enc_attn_weights_vit = []

    # Recover a displayable image robustly across processors; fall back to original when needed.
    try:
        img_std = torch.tensor(processor.image_processor.image_std).view(3,1,1)
        img_mean = torch.tensor(processor.image_processor.image_mean).view(3,1,1)
        img_recover = inputs.pixel_values[0].cpu() * img_std + img_mean
        img_recover = to_pil_image(img_recover)
    except Exception:
        img_recover = state.image if isinstance(state.image, Image.Image) else image

    state.recovered_image = img_recover
    state.input_text_tokenized = input_text_tokenized
    state.output_ids_decoded = output_ids_decoded 
    state.attention_key = tempfilename.name
    state.image_idx = img_idx
    
    # Persist OpenVLA flags and derived grid layout info for cross-attn visualization.
    state.is_openvla = is_openvla
    if is_openvla and cross_attn_obj is not None:
        try:
            kv_len = cross_attn_obj[0][0].shape[-1]
            import math
            g = int(math.sqrt(kv_len))
            ignore_first = 0
            if g * g != kv_len and (kv_len - 1) > 0:
                g2 = int(math.sqrt(kv_len - 1))
                if g2 * g2 == (kv_len - 1):
                    g = g2
                    ignore_first = 1
            state.enc_grid_rows = g
            state.enc_grid_cols = g
            state.enc_kv_ignore_first = ignore_first
        except Exception:
            state.enc_grid_rows = 24
            state.enc_grid_cols = 24
            state.enc_kv_ignore_first = 0

    return state, to_gradio_chatbot(state) 


def build_demo(args, embed_mode=False):
    global model
    global processor
    global system_prompt
    global ROLE0
    global ROLE1

    if model is None:
        processor, model = get_processor_model(args)

    if 'gemma' in args.model_name_or_path:
        system_prompt = ''
        ROLE0 = 'user'
        ROLE1 = 'model'

    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LVLM-Interpret", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column(scale=6):
                    
                    imagebox = gr.ImageEditor(type="pil", height=400, elem_id="image_canvas")
                    
                    
                    with gr.Accordion("Parameters", open=False) as parameter_row:
                        image_process_mode = gr.Radio(
                            ["Crop", "Resize", "Pad", "Default"],
                            value="Default",
                            label="Preprocess for non-square image", visible=True
                        )
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                        max_output_tokens = gr.Slider(minimum=0, maximum=512, value=32, step=32, interactive=True, label="Max new output tokens",)


                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(elem_id="chatbot", label="Chatbot", height=400)
                    with gr.Row():
                        with gr.Column(scale=8):
                            textbox.render()
                        with gr.Column(scale=1, min_width=50):
                            submit_btn = gr.Button(value="Send", variant="primary")
                    with gr.Row(elem_id="buttons") as button_row:
                        clear_btn = gr.Button(value="🗑️  Clear", interactive=True, visible=True)

        with gr.Tab("Attention analysis"):
            with gr.Row():
                with gr.Column(scale=3):
                    attn_modality_select = gr.Dropdown(
                            choices=['Image-to-Answer', 'Question-to-Answer'],
                            value='Image-to-Answer',
                            interactive=True,
                            show_label=False,
                            container=False
                        )
                    attn_ana_submit = gr.Button(value="Plot attention matrix", interactive=True)
                with gr.Column(scale=6):
                    attn_ana_plot = gr.Plot(label="Attention plot")

        attn_ana_submit.click(
                plot_attention_analysis,
                [state, attn_modality_select],
                [state, attn_ana_plot]
            )

        with gr.Tab("Attentions"):
            with gr.Row():
                attn_select_layer = gr.Slider(1, N_LAYERS, value=32, step=1, label="Layer")
            with gr.Row():
                with gr.Column(scale=3):
                    imagebox_recover = gr.Image(type="pil", label='Preprocessed image', interactive=False)

                    generated_text = gr.HighlightedText(
                        label="Generated text (tokenized)",
                        combine_adjacent=False,
                        interactive=True,
                        color_map={"label": "green"}
                    )
                    with gr.Row():
                        attn_reset = gr.Button(value="Reset tokens", interactive=True)
                        attn_submit = gr.Button(value="Plot attention", interactive=True)

                with gr.Column(scale=9):
                    i2t_attn_head_mean_plot = gr.Plot(label="Image-to-Text attention average per head")
                    i2t_attn_gallery = gr.Gallery(type="pil", label='Attention heatmaps', columns=8, interactive=False)

            box_states = gr.Dataframe(type="numpy", datatype="bool", row_count=24, col_count=24, visible=False) 
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    imagebox_recover_boxable = gr.Image(label='Patch Selector')
                    attn_ana_head= gr.Slider(1, 40, step=1, label="Head Index")
            
                    reset_boxes_btn = gr.Button(value="Reset patch selector")
                    attn_ana_submit_2 = gr.Button(value="Plot attention matrix", interactive=True)
                
                with gr.Column(scale=9):
                    t2i_attn_head_mean_plot = gr.Plot(label="Text-to-Image attention average per head")
                    attn_ana_plot_2 = gr.Plot(scale=2, label="Attention plot",container=True)

        reset_boxes_btn.click(
            handle_box_reset, 
            [imagebox_recover,box_states], 
            [imagebox_recover_boxable, box_states]
        )
        imagebox_recover_boxable.select(boxes_click_handler, [imagebox_recover,box_states], [imagebox_recover_boxable, box_states])
        
        attn_reset.click(
            reset_tokens,
            [state],
            [generated_text]
        )

        attn_ana_submit_2.click(
            plot_text_to_image_analysis,
            [state, attn_select_layer, box_states, attn_ana_head ],
            [state, attn_ana_plot_2, t2i_attn_head_mean_plot]
        )
        
        
        attn_submit.click(
            handle_attentions_i2t,
            [state, generated_text, attn_select_layer],
            [generated_text, imagebox_recover, i2t_attn_gallery, i2t_attn_head_mean_plot]
        )

        with gr.Tab("Relevancy"):
            with gr.Row():
                relevancy_token_dropdown = gr.Dropdown(
                    choices=['llama','vit','all'],
                    value='llama',
                    interactive=True,
                    show_label=False,
                    container=False
                )
                relevancy_submit = gr.Button(value="Plot relevancy", interactive=True)
            with gr.Row():
                relevancy_gallery = gr.Gallery(type="pil", label='Input image relevancy heatmaps', columns=8, interactive=False)
            with gr.Row():
                relevancy_txt_gallery = gr.Gallery(type="pil", label='Image-text relevancy comparison', columns=8, interactive=False)
            with gr.Row():
                relevancy_highlightedtext = gr.HighlightedText(
                        label='Tokens with high relevancy to image'
                    )

        relevancy_submit.click(
            lambda state, relevancy_token_dropdown: handle_relevancy(state, relevancy_token_dropdown, incude_text_relevancy=True),
            [state, relevancy_token_dropdown],
            [relevancy_gallery],
        )
        relevancy_submit.click(
            handle_text_relevancy,
            [state, relevancy_token_dropdown],
            [relevancy_txt_gallery, relevancy_highlightedtext]
        )

        enable_causality = False
        with gr.Tab("Causality"):
            gr.Markdown(
                """
                ### *Coming soon*
                """
            )
            state_causal_explainers = gr.State()
            with gr.Row(visible=enable_causality):
                causality_dropdown = gr.Dropdown(
                    choices=[],
                    interactive=True,
                    show_label=False,
                    container=False,
                    scale=2,
                )
                causality_submit = gr.Button(value="Learn Causal Structures", interactive=True, variant='primary', scale=1)
            with gr.Row(visible=enable_causality):
                with gr.Accordion("Hyper Parameters", open=False) as causal_parameters_row:
                        with gr.Row():
                            with gr.Column(scale=2):
                                att_th_slider = gr.Slider(minimum=0.0001, maximum=1-0.0001, value=0.005, step=0.0001, interactive=True, label="Raw Attention Threshold",
                                                          info="A threshold for selecting tokens to be graph nodes.",)
                            with gr.Column(scale=2):
                                alpha_slider = gr.Slider(minimum=1e-7, maximum=1e-2, value=1e-5, step=1e-7, interactive=True, label="Statistical Test Threshold (alpha)",
                                                         info="A threshold for the statistical test of conditional independence.",)
            with gr.Row(visible=enable_causality):
                pds_plot = gr.Image(type="pil", label='Preprocessed image')
                causal_head_gallery = gr.Gallery(type="pil", label='Causal Head Graph', columns=8, interactive=False)
            with gr.Row(visible=enable_causality):
                causal_head_slider = gr.Slider(minimum=0, maximum=31, value=1, step=1, interactive=True, label="Head Selection")
                causal_head_submit = gr.Button(value="Plot Causal Head", interactive=True, scale=1)
            with gr.Row(visible=enable_causality):
                causality_gallery = gr.Gallery(type="pil", label='Causal Heatmaps', columns=8, interactive=False)
    
        causal_head_submit.click(
            handle_causal_head,
            [state, state_causal_explainers, causal_head_slider, causality_dropdown],
            [causal_head_gallery, pds_plot]
        )
        
        causality_submit.click(
            handle_causality,
            [state, state_causal_explainers, causality_dropdown, alpha_slider, att_th_slider],
            [causality_gallery, state_causal_explainers]
        )

        if not embed_mode:
            gr.Markdown(tos_markdown)

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox, imagebox_recover, generated_text, i2t_attn_gallery ] ,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            lvlm_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] ,
        ).then(
            attn_update_slider,
            [state],
            [state, attn_select_layer]
        ).then(
            causality_update_dropdown,
            [state],
            [state, causality_dropdown]
        )
        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            lvlm_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot],
        ).then(
            attn_update_slider,
            [state],
            [state, attn_select_layer]
        ).then(
            causality_update_dropdown,
            [state],
            [state, causality_dropdown]
        )
        

    return demo

 