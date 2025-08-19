import sys
sys.path.append('causality_lab')

import logging
import os
import numpy as np
import gradio as gr
import torch
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
from plot_utils import draw_graph, draw_pds_tree
from causal_discovery_utils.cond_indep_tests import CondIndepParCorr

logger = logging.getLogger(__name__)

from utils_causal_discovery_fn import (
    get_relevant_image_tokens,
    tokens_analysis,
    create_explanation,
    copy_sub_graph,
    show_tokens_on_image,
    calculate_explanation_pvals,
    get_relevant_prompt_tokens,
    get_relevant_text_tokens,
    crop_token,
    get_expla_set_per_rad
)


def create_im_tokens_marks(orig_img, tokens_to_mark, weights=None, txt=None, txt_pos=None, n_x_tokens: int = 24, n_y_tokens: int = 24):
    im_1 = orig_img.copy()
    if weights is not None:
        im_heat = show_tokens_on_image(tokens_to_mark, im_1, weights, n_x_tokens=n_x_tokens, n_y_tokens=n_y_tokens)
    else:
        im_heat = show_tokens_on_image(tokens_to_mark, im_1, n_x_tokens=n_x_tokens, n_y_tokens=n_y_tokens)
    im_heat_edit = ImageDraw.Draw(im_heat)
    if isinstance(txt, str):
        if txt_pos is None:
            txt_pos = (10, 10)
        im_heat_edit.text(txt_pos, txt, fill=(255, 255, 255))
    im_heat = im_heat_edit._image
    return im_heat


def causality_update_dropdown(state):
    generated_text = state.output_ids_decoded
    choices = [ f'{i}_{tok}' for i,tok in enumerate(generated_text)]
    return state, gr.Dropdown(value=choices[0], interactive=True, scale=2, choices=choices)


def handle_causal_head(state, explainers_data, head_selection, class_token_txt):
    recovered_image = state.recovered_image
    is_openvla = getattr(state, 'is_openvla', False)
    # Define image token start and dynamic grid sizes
    first_im_token_idx = 0 if is_openvla else state.image_idx
    g_rows = getattr(state, 'enc_grid_rows', 24) or 24
    g_cols = getattr(state, 'enc_grid_cols', 24) or 24
    N_img_local = (g_rows * g_cols) if is_openvla else 576
    first_im_token_idx = 0 if is_openvla else state.image_idx
    g_rows = getattr(state, 'enc_grid_rows', 24)
    g_cols = getattr(state, 'enc_grid_cols', 24)
    num_image_tokens = (g_rows * g_cols) if is_openvla else 576

    token_to_explain = explainers_data[0]
    head_id = head_selection
    explainer = explainers_data[1][head_id]
    if explainer is None:
        return [], None

    expla_set_per_rad = get_expla_set_per_rad(explainer.results[token_to_explain]['pds_tree'])
    max_depth = max(expla_set_per_rad.keys())
    im_heat_list = []
    im_tok_rel_idx = []
    for rad in range(1,max_depth+1):
        im_tok_rel_idx += [v-first_im_token_idx 
                           for v in expla_set_per_rad[rad] if v >= first_im_token_idx and v < (first_im_token_idx+num_image_tokens)]
        im_heat_list.append(
            create_im_tokens_marks(recovered_image, im_tok_rel_idx, txt='search radius: {rad}'.format(rad=rad), n_x_tokens=g_rows, n_y_tokens=g_cols)
        )
        

    # im_graph_list = []
    # for r in range(1, 5):
    #     expla_list = explainer.explain(token_to_explain, max_range=r)[0][0]
    #     nodes_set = set(expla_list)
    #     nodes_set.add(token_to_explain)
    #     subgraph = copy_sub_graph(explainer.graph, nodes_set)
    #     fig = draw_graph(subgraph, show=False)
    #     fig.canvas.draw()
    #     im_graph = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    #     plt.close()
    #     im_graph_list.append(im_graph)

    expla_list = explainers_data[2][head_id]
    node_labels = dict()
    for tok in expla_list:
        im_idx = tok - first_im_token_idx
        if im_idx < 0 or im_idx >= num_image_tokens:  # if token is not image
            continue
        im_tok = crop_token(recovered_image, im_idx, pad=2)
        node_labels[tok] = im_tok.resize((45, 45))

    node_labels[token_to_explain] = class_token_txt.split('_')[1]
    
    nodes_set = set(expla_list)
    nodes_set.add(token_to_explain)
    # Coerce labels to safe types (stringify unexpected objects)
    node_labels = {k: (v if isinstance(v, (str, int, Image.Image)) else str(v)) for k, v in node_labels.items()}
    fig = draw_pds_tree(explainer.results[token_to_explain]['pds_tree'], explainer.graph, node_labels=node_labels,
                        node_size_factor=1.4)
    if fig is None:
        fig = plt.figure()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    im_graph = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1).convert('RGB')
    plt.close()

    return im_heat_list, im_graph


def handle_causality(state, state_causal_explainers, token_to_explain, alpha_ext=None, att_th_ext=None):
    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Results' containers ---***---
    gallery_image_list = []
    gallery_graph_list = []
    gallery_bar_graphs = []
    
    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Generic app handling ---***---
    if not hasattr(state, 'attention_key'):
        return []
    
    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Load attention matrix ---***---
    fn_attention = state.attention_key + '_attn.pt'
    fn_xattention = state.attention_key + '_xattn.pt'
    recovered_image = state.recovered_image
    generated_text = state.output_ids_decoded

    is_openvla = getattr(state, 'is_openvla', False)
    if not os.path.exists(fn_attention) and (not (is_openvla and os.path.exists(fn_xattention))):
        gr.Error('Attention file not found. Please re-run query.')
    else:
        # Torch 2.6 default weights_only=True breaks list-of-tensors; disable it
        attentions = torch.load(fn_attention, weights_only=False) if os.path.exists(fn_attention) else None

    if not is_openvla:
        last_mh_attention = attentions[-1][-1]
        num_heads, _, attention_len = last_mh_attention[-1].shape
        full_attention = np.zeros((num_heads, attention_len, attention_len))

        last_mh_attention = attentions[0][-1]
        attention_vals = last_mh_attention[0].detach().cpu().numpy()
        d1 = attention_vals.shape[-1]
        full_attention[:, :d1, :d1] = attention_vals

    # create one full attention matrix that includes attention to generated tokens
        for gen_idx in range(1, len(generated_text)):
            last_mh_attention = attentions[gen_idx][-1]
            att_np = last_mh_attention[0].detach().cpu().numpy()
            full_attention[:, d1, :att_np.shape[-1]] = att_np[:,0,:]
            d1 += 1

    # OpenVLA: build a square matrix combining cross-attn (text->image) and decoder self-attn (text->text)
    if is_openvla:
        xattn = torch.load(fn_xattention, weights_only=False) if os.path.exists(fn_xattention) else None
        if xattn is None:
            return []
        g_rows = getattr(state, 'enc_grid_rows', 24) or 24
        g_cols = getattr(state, 'enc_grid_cols', 24) or 24
        ignore_first = getattr(state, 'enc_kv_ignore_first', 0)
        N_img = g_rows * g_cols
        N_txt = len(generated_text)
        N_txt_eff = min(N_txt, len(xattn))
        N = N_img + N_txt_eff
        # Use the last layer
        # Determine heads from first token
        first_tok_last = xattn[0][-1].squeeze()  # [H,Q,K] or [H,K]
        if first_tok_last.dim() == 1:
            first_tok_last = first_tok_last.unsqueeze(0)
        if first_tok_last.dim() == 3:
            first_tok_last = first_tok_last[:, -1, :]
        num_heads = first_tok_last.shape[0]
        full_attention = np.zeros((num_heads, N, N))
        # Bottom-left: text->image from cross-attn
        for t in range(N_txt_eff):
            mh = xattn[t][-1].squeeze()
            if mh.dim() == 3:
                mh = mh[:, -1, :]
            # slice K to grid window
            kv = mh[:, ignore_first:ignore_first + N_img]
            kv = kv[:, :N_img]
            full_attention[:, N_img + t, :N_img] = kv.detach().cpu().numpy()
        # Bottom-right: text->text from decoder self-attn if available, else small identity
        if attentions is not None:
            try:
                dec_last = attentions[0][-1].squeeze()  # [H,Q,K]
                if dec_last.dim() == 2:
                    dec_last = dec_last.unsqueeze(1)
                Q, K = dec_last.shape[-2], dec_last.shape[-1]
                q_start = max(0, Q - N_txt_eff)
                k_start = max(0, K - N_txt_eff)
                for h in range(num_heads):
                    block = dec_last[h, q_start:, k_start:].detach().cpu().numpy()
                    Hq, Hk = block.shape
                    full_attention[h, N_img:N_img+Hq, N_img:N_img+Hk] = block
            except Exception:
                for h in range(num_heads):
                    full_attention[h, N_img:N, N_img:N] = np.eye(N_txt_eff) * 1e-3
        else:
            for h in range(num_heads):
                full_attention[h, N_img:N, N_img:N] = np.eye(N_txt_eff) * 1e-3
        # Top-left: image->image unknown; set weak identity
        for h in range(num_heads):
            full_attention[h, :N_img, :N_img] = np.eye(N_img) * 1e-6

    # Sizes:
    # Number of heads: {num_heads}, attention size: {attention_len}x{attention_len}

    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Hyper-parameters for causal discovery ---***---
    threshold = 1e-5  # alpha; threshold for p-value in conditional independence testing
    degrees_of_freedom = 128
    default_search_range = 3
    max_num_image_tokens = 50  # number of image-tokens to consider as 'observed'. Used for calculating head importance
    att_th = 0.01  # threshold for attention values. Below this value, the token is considered 'not-attented'
    search_range = default_search_range  # causal-explanation seach-distance in the causal graph
    if alpha_ext is not None:
        threshold = alpha_ext
    if att_th_ext is not None:
        att_th = att_th_ext

    # Derive consistent indexing helpers for both backends
    g_rows = getattr(state, 'enc_grid_rows', 24) or 24
    g_cols = getattr(state, 'enc_grid_cols', 24) or 24
    N_img_local = (g_rows * g_cols) if is_openvla else 576
    first_im_token_idx = 0 if is_openvla else state.image_idx

    heads_to_analyse = list(range(num_heads))
    if not is_openvla:
        token_to_explain = attention_len - len(generated_text) + int(token_to_explain.split('_')[0])
    else:
        idx_local = int(token_to_explain.split('_')[0])
        idx_local = min(max(0, idx_local), max(0, N_txt_eff - 1))
        token_to_explain = N_img + idx_local
    logger.info(f'Using token index {token_to_explain} for explaining')

    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Learn causal Structure ---***---

    time_struct = []  # list of runtime results for learning the structure for different heads
    time_reason = []  # list of runtime results for recovering an explanation for different heads

    expla_list_all = [None] * num_heads
    explainer_all = [None] * num_heads
    timing_all = [None] * num_heads
    head_importance = [0] * num_heads

    # state_causal_explainers[0] = token_to_explain
    # state_causal_explainers[1] = []
    state_causal_explainers = [token_to_explain]
    # state_causal_explainers.append(dict())

    total_weights = [0 for _ in range(N_img_local)]  # weights for image tokens

    for head_id in heads_to_analyse:  # ToDo: Run in parallel (threading/multiprocessing; a worker for head)
        head_attention = full_attention[head_id]  # alias for readability

        #  ---***------***--- Text causal graph ---***------***---
        start_txt = (N_img_local if is_openvla else first_im_token_idx+576)
        text_expla, text_expl, timing = tokens_analysis(
            head_attention,
            list(range(start_txt, token_to_explain+1)),
            token_of_interest=token_to_explain,
            number_of_samples=degrees_of_freedom,
            p_val_thrshold=threshold,
            max_search_range=search_range,
            verbose=False,
        )
        txt_node_labels = dict()
        for v in text_expla:
            # Map text node index back to token
            if not is_openvla:
                idx = v - (attention_len - len(generated_text))
            else:
                idx = v - N_img_local
            if 0 <= idx < len(generated_text):
                txt_node_labels[v] = generated_text[idx]
        #  End: *------***--- Text causal graph ---***------***---
        

        w = head_attention[token_to_explain, :]
        w_img = (w[0:N_img_local] if is_openvla else w[first_im_token_idx:(first_im_token_idx+576)])
        # im_entropy = -np.nansum(w_img * np.log(w_img))
        # total_entropy = -np.nansum(w * np.log(w))
        
        # print(f'{head_id}: total_entropy: {total_entropy}, image entropy: {im_entropy}, entropy diff: im - total: {im_entropy - total_entropy}')
        num_high_att = max(2, sum(w > att_th))

        num_image_tokens = min(num_high_att, max_num_image_tokens)  # number of image tokens to select for analysis

        relevant_image_idx = get_relevant_image_tokens(
            class_token=token_to_explain,
            attention_matrix=head_attention,
            first_token=(0 if is_openvla else first_im_token_idx),
            num_top_k_tokens=num_image_tokens,
            num_image_tokens=N_img_local,
        )

        relevant_gen_idx = get_relevant_text_tokens(
            class_token=token_to_explain,
            attention_matrix=head_attention,
            att_th=att_th,
            first_image_token=(0 if is_openvla else first_im_token_idx),
            num_image_tokens=N_img_local,
        )
        relevant_tokens = relevant_image_idx + relevant_gen_idx + [token_to_explain]

        # print(f'Self: {head_attention[token_to_explain, token_to_explain]}')
        # att_th = head_attention[token_to_explain, token_to_explain]
        # att_th = np.median(w[first_im_token_idx+576:])
        # print(f'Attentnion threshold: {att_th}')
        # relevant_tokens = set(np.where(w >= att_th)[0])
        # relevant_tokens.add(token_to_explain)
        # relevant_tokens = list(relevant_tokens)
        # relevant_tokens = [v for v in relevant_tokens if v >= first_im_token_idx]
        # print('relevant tokens', relevant_tokens)

        expla_list, explainer, timing = tokens_analysis(head_attention, relevant_tokens,
                                                        token_of_interest=token_to_explain,
                                                        number_of_samples=degrees_of_freedom, p_val_thrshold=threshold, max_search_range=search_range, 
                                                        verbose=False)

        expla_list_all[head_id] = expla_list
        explainer_all[head_id] = explainer
        timing_all[head_id] = timing

        # calculate Head Importance
        im_expla_tokens_list = [
            v for v in expla_list
            if (v >= (0 if is_openvla else first_im_token_idx))
            and (v < ((0 if is_openvla else first_im_token_idx) + N_img_local))
        ]  # only image explanation
        ci_test = explainer.ci_test
        prev_num_records = ci_test.num_records
        ci_test.num_records = len(im_expla_tokens_list)
        weights_list = []
        for im_expla_tok in im_expla_tokens_list:
            cond_set = tuple(set(im_expla_tokens_list) - {im_expla_tok})
            p_val = min(ci_test.calc_statistic(im_expla_tok, token_to_explain, cond_set), 1)  # avoid inf
            weights_list.append(1-p_val)
        ci_test.num_records = prev_num_records

        # print(f'*** Head: {head_id} -- weights: {weights_list}')
        # if len(weights_list) == 0:
        #     head_importance[head_id] = 0
        # else:
        #     head_importance[head_id] = np.mean(weights_list)
        denom_start = (N_img_local if is_openvla else first_im_token_idx+576)
        head_importance[head_id] = max(w_img) / max(w[denom_start:])

        for im_expla_tok, im_expla_weight in zip(im_expla_tokens_list, weights_list):
            total_weights[im_expla_tok-first_im_token_idx] += im_expla_weight

        # if len(im_expla_tokens_list) > 0:
        #     head_importance[head_id] = np.median(w[im_expla_tokens_list])
        # else:
        #     head_importance[head_id] = 0

        # p_vals_dict = calculate_explanation_pvals(explainer, token_to_explain, search_range)
        # p_weights_im_tokens = [
        #     (1-p_vals_dict[v])*w[v] for v in expla_list if (v >= first_im_token_idx) and (v < first_im_token_idx + 576)
        # ]
        # if len(p_weights_im_tokens) == 0:
        #     head_importance[head_id] = 0
        # else:
        #     head_importance[head_id] = np.median(p_weights_im_tokens)

        # if len(expla_list) > 0:
        #     # head_importance[head_id] = np.median(w[expla_list])
        #     head_importance[head_id] = np.median(sorted(w)[-max_num_image_tokens:])
        # else:
        #     head_importance[head_id] = 0
            
        txt = '{head}:    {importance:.2f}    / 100'.format(head=head_id, importance=head_importance[head_id]*100)
        logger.info(f'Head: {head_id}: importance: {txt}')

        
        time_struct.append(timing['structure'])
        time_reason.append(timing['reasoning'])
        im_expla_rel_idx = [v-first_im_token_idx for v in im_expla_tokens_list]  # only image

        # print(f'head {head_id}, importance: {head_importance[head_id]:.3f}, above {att_th}: {num_high_att}')

        # plot results
        logger.info('Max: *******', max(total_weights))
        if max(total_weights) > 0:
            norm_total_weights =  [v/max(total_weights) for v in total_weights]
        else:
            norm_total_weights = total_weights
        im_t = recovered_image.copy()
        if is_openvla:
            im_heat_total = show_tokens_on_image(
                list(range(N_img_local)), im_t, norm_total_weights, n_x_tokens=g_rows, n_y_tokens=g_cols
            )
        else:
            im_heat_total = show_tokens_on_image(list(range(576)), im_t, norm_total_weights)
        im_heat_edit_t = ImageDraw.Draw(im_heat_total)
        im_heat_edit_t.text((10, 10), txt, fill=(255, 255, 255))
        im_heat_total = im_heat_edit_t._image

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(range(num_heads), head_importance)
        ax.grid(True)
        xmin, xmax, ymin, ymax = ax.axis()
        ax.axis([1, 32, -ymax*0.01, ymax])
        # ax.set_position([0, 0, 1, 1])
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        im_head_importance = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1).convert('RGB')
        plt.close()

        # attentnion values
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if is_openvla:
            h = [max(w[0:N_img_local])] + list(w[N_img_local:])
        else:
            h = [max(w[first_im_token_idx:576+first_im_token_idx])] + list(w[first_im_token_idx+576:])
        ax.bar(range(len(h)), h)
        ax.grid(True)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        im_att_bar = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1).convert('RGB')
        plt.close()

        im_heat = create_im_tokens_marks(recovered_image, im_expla_rel_idx, txt=txt)
        # im_1 = recovered_image.copy()
        # # im_heat = show_tokens_on_image(im_expla_rel_idx, im_1, weights_list)
        # im_heat = show_tokens_on_image(im_expla_rel_idx, im_1)
        # im_heat_edit = ImageDraw.Draw(im_heat)
        # im_heat_edit.text((10, 10), txt, fill=(255, 255, 255))
        # im_heat = im_heat_edit._image

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(head_importance, '.-')
        ax.grid(True)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        im_pl = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1).convert('RGB')
        plt.close()

        nodes_set = set(expla_list)
        nodes_set.add(token_to_explain)
        subgraph = copy_sub_graph(explainer.graph, nodes_set)
        fig = draw_graph(subgraph, show=False)
        fig.canvas.draw()
        # im_graph = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        plt.close()

        # nodes_set = set(text_expla)
        # nodes_set.add(token_to_explain)
        # subgraph = copy_sub_graph(text_expl.graph, nodes_set)
        # fig = draw_graph(subgraph, show=False, node_labels=node_labels)
        # fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        im_graph = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1).convert('RGB')
        # plt.close()

        node_labels = dict()
        for tok in expla_list:
            if tok in txt_node_labels:  # if token is text
                node_labels[tok] = txt_node_labels[tok]
                continue
            im_idx = tok - (0 if is_openvla else first_im_token_idx)
            if im_idx < 0 or im_idx >= N_img_local:  # if token is not image
                continue
            im_tok = crop_token(recovered_image, im_idx, pad=2)
            node_labels[tok] = im_tok.resize((45, 45))

        if not is_openvla:
            idx = token_to_explain - (attention_len - len(generated_text))
        else:
            idx = token_to_explain - N_img_local
        node_labels[token_to_explain] = generated_text[idx]
        # Coerce labels to safe types
        node_labels = {k: (v if isinstance(v, (str, int, Image.Image)) else str(v)) for k, v in node_labels.items()}
        
        nodes_set = set(expla_list)
        nodes_set.add(token_to_explain)
        fig = draw_pds_tree(explainer.results[token_to_explain]['pds_tree'], explainer.graph, node_labels=node_labels,
                          node_size_factor=1.4)
        if fig is None:
            fig = plt.figure()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        im_graph = Image.frombytes('RGBA', fig.canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1).convert('RGB')
        plt.close()

        gallery_image_list.append(im_heat)
        gallery_graph_list.append(im_graph)
        gallery_bar_graphs.append(im_att_bar)
        # gallery_image_list.append(im_pl)
    
    state_causal_explainers.append(explainer_all)  # idx 1
    state_causal_explainers.append(expla_list_all) # idx 2
    return gallery_image_list + gallery_graph_list + gallery_bar_graphs, state_causal_explainers #im_heat_total #im_head_importance
