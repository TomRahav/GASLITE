import torch
from typing import List, Tuple, Union, Dict
import numpy as np


def token_gradients(
    model,  # Targeted HuggingFace model
    trigger_slice: slice,  # targeted slice
    inputs: Dict[str, torch.Tensor],  # with input_ids, (batch_size, seq_len)
    input_embedding_layer: torch.nn.Embedding,  # embedding layer
    device: str = "cuda",
    # Fluency constraints
    flu_model=None,
    flu_alpha=0.0,
    l2_alpha=0.0,
    **kwargs,  # additional kwargs to pass to the model for calculating the loss (e.g., labels)
):
    """
    Returns the gradient with respect to the one-hot trigger tokens in `trigger_slice`,
    computed in a memory-efficient way using torch.autograd.grad so that only the one_hot
    gradients are kept in memory. The fluency and L2 penalty terms are added in every iteration.
    """
    method = kwargs.get("chunk_robustness_method", None)
    input_ids = inputs["input_ids"]
    original_stop = trigger_slice.stop
    trigger_len = original_stop - trigger_slice.start
    iterations = trigger_len if "avg_loss" in method else 1

    grad_accum = None  # will accumulate per-iteration gradients

    for i in range(iterations):
        torch.cuda.empty_cache()
        current_slice = slice(trigger_slice.start, original_stop - i)
        current_trigger_len = trigger_len - i
        embed_weights = input_embedding_layer.weight
        targeted_seq_len = input_ids[:, current_slice].shape[1]
        one_hot = torch.zeros(
            input_ids.shape[0],  # batch_size
            targeted_seq_len,  # length of targeted subsequence
            embed_weights.shape[0],  # vocab_size
            device=device,
            dtype=embed_weights.dtype,
        )
        one_hot.scatter_(
            -1,
            input_ids[:, current_slice].unsqueeze(-1),
            1.0,
        )
        one_hot.requires_grad_()
        # Compute trigger embeddings using the one-hot representation.
        trigger_embeds = (
            one_hot @ embed_weights
        )  # (batch_size, targeted_seq_len, embed_dim)
        embeds = input_embedding_layer(input_ids).detach()
        full_embeds = torch.cat(
            [
                embeds[:, : trigger_slice.start, :],
                trigger_embeds,
                embeds[:, current_slice.stop :, :],
            ],
            dim=1,
        )

        # Calculate the base loss from the model.
        loss = model.calc_loss_for_grad(
            inputs_embeds=full_embeds,
            inputs_attention_mask=inputs["attention_mask"],
            **kwargs,
        )
        if i == 0:
            # Add the fluency penalty in each iteration, if applicable.
            if flu_alpha != 0 and flu_model is not None:
                fluency_score = flu_model.calc_score_for_grad(
                    one_hot=one_hot,
                    trigger_slice=current_slice,
                    inputs=inputs,
                )
                loss = loss + flu_alpha * fluency_score

            # Add the L2 penalty in each iteration, if applicable.
            if l2_alpha != 0:
                loss = loss - l2_alpha * trigger_embeds.norm(dim=-1).sum()

        weighted_loss = loss / (i + 1)
        # Compute gradient only with respect to one_hot.
        grad_i = torch.autograd.grad(weighted_loss, one_hot, retain_graph=False)[0]
        if "onehot_grad" in method:
            mask = torch.zeros_like(grad_i)
            mask[:, -1, :] = 1
            grad_i = grad_i * mask
        elif "weighted_linearly" in method:
            # Given L~Uni(1,T), P(L>=j | L>=i and j>=i) = 1 - P(L < j | L>=i and j>=i) =
            # L*~Uni(i, T) 1 - P(L*<j) = 1 - ((j-i) / (T - (i-1))) = (T - i + 1 - j + i) / (T - (i-1))) = (T - (j-1)) / (T - (i-1))
            length_plus_one = (
                torch.ones_like(grad_i) + torch.ones_like(grad_i) * trigger_len
            )
            mask = (length_plus_one - torch.ones_like(grad_i) * current_trigger_len) / (
                length_plus_one
                - torch.arange(
                    start=1, end=current_trigger_len + 1, device=grad_i.device
                ).view(1, current_trigger_len, 1)
            )
            grad_i = grad_i * mask
        # Pad grad_i with zeros so that its length matches full_trigger_len.
        if grad_accum is None:
            grad_accum = grad_i
        else:
            pad_len = trigger_len - current_trigger_len
            pad = torch.zeros(
                grad_i.shape[0],
                pad_len,
                grad_i.shape[2],
                device=device,
                dtype=grad_i.dtype,
            )
            grad_i = torch.cat([grad_i, pad], dim=1)
            grad_accum = grad_accum + grad_i

        # Clean up intermediate tensors.
        del one_hot, trigger_embeds, embeds, full_embeds, loss, weighted_loss, grad_i
        torch.cuda.empty_cache()

    onehot_grads = grad_accum / grad_accum.norm(dim=-1, keepdim=True)
    return onehot_grads


def get_trigger_candidates(
    scores: torch.Tensor,  # scores of all possible tokens, highest-better (targeted_sub_seq_len, vocab_size)
    trigger_slice: slice,  # targeted slice
    k_candidates: int = 5,  # number of candidates to return
    candidate_scheme: str = "best_per_token",  # 'best_per_token' | 'best_overall'
    token_ids_to_ignore: List[
        int
    ] = None,  # usually the special tokens (which we don't want to consider)
    flu_scores: torch.Tensor = None,  # fluency scores of all possible tokens, highest-better (seq_len, vocab_size)
    filter_to_n_most_readable: int = 700,
) -> Union[Dict[int, Dict[str, List[Union[int, float]]]], List[Tuple[int, int]]]:
    """
    Translates the scores to the actual tokens to flip to.
    :returns: the `num_candidates` best tokens to flip to in each position in the `trigger_slice`; the list is sorted
              from the best token position to flip, to the rest.
              The returns list has each element of (token_idx_to_flip, token_id_to_flip_to).
    Note! this function is indented for a _single_ sample.
    """
    candidates = {}  # {token_idx -> (token_top_ids, token_top_scores), ...}
    scores = scores.clone()
    scores[:, token_ids_to_ignore] = (
        -np.inf
    )  # ignore the special tokens, set their scores to the lowest possible
    if candidate_scheme == "top_k_per_token":
        for token_idx_in_trigger, token_idx in enumerate(
            range(trigger_slice.start, trigger_slice.stop)
        ):
            if flu_scores is not None and filter_to_n_most_readable is not None:
                # ignore tokens with low readability
                indices_with_low_readability = np.argsort(
                    flu_scores[token_idx_in_trigger].cpu()
                )[:-filter_to_n_most_readable]
                scores[token_idx_in_trigger, indices_with_low_readability] = -np.inf
            token_top_k_obj = scores[token_idx_in_trigger].topk(k_candidates)
            token_top_ids = token_top_k_obj.indices.tolist()
            token_top_scores = token_top_k_obj.values.tolist()
            candidates[token_idx] = dict(
                token_top_ids=token_top_ids, token_top_scores=token_top_scores
            )
    elif candidate_scheme == "top_k_overall":  # Disabled
        v, i = torch.topk(scores.flatten(), k_candidates)
        v, i = v.cpu(), i.cpu()
        candidates = np.array(np.unravel_index(i.numpy(), scores.shape)).T
        candidates[:, 0] += trigger_slice.start  # shift indices to the correct position
        candidates = [(token_idx, token_id) for token_idx, token_id in candidates]
    else:
        raise ValueError(f"Unknown candidate scheme: {candidate_scheme}")
    return candidates


def get_tokenized_sliced_sentence(tokenized, i, slice_order, trigger_len):
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    # Identify the length of the valid tokens (excluding padding)
    valid_token_count = attention_mask.sum(dim=1).item()

    # Ensure we only truncate from the valid tokens
    if valid_token_count > i:
        eos = input_ids[:, valid_token_count - 1 : valid_token_count]
        info_len = valid_token_count - trigger_len
        info_input_ids = input_ids[:, :info_len]
        # Truncate valid tokens
        if slice_order == "end":
            truncated_input_ids = input_ids[:, : valid_token_count - i - 1]
            truncated_input_ids = torch.cat((truncated_input_ids, eos), dim=1)
        elif slice_order == "start":
            truncated_trigger = input_ids[:, info_len + i :]
            truncated_input_ids = torch.cat((info_input_ids, truncated_trigger), dim=1)
        elif slice_order == "shuffle":
            shuffeled_trigger = input_ids[:, info_len + i :]
            # Shuffle along the last dimension (columns) for each row
            indices = torch.stack(
                [
                    torch.randperm(shuffeled_trigger.shape[1])
                    for _ in range(shuffeled_trigger.shape[0])
                ]
            )
            shuffled_input_ids = torch.gather(input_ids, dim=1, index=indices)
            truncated_input_ids = torch.cat(
                (info_input_ids, shuffled_input_ids, eos), dim=1
            )
    return truncated_input_ids
