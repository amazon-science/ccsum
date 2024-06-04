from bert_score import score
import torch
import numpy as np
import tqdm


def evaluate_bert_score(
    cands, refs, device="cuda", model_type="bert-large-uncased", batch_size=32
):
    with torch.no_grad():
        # P, R, F1 = score(cands, refs, lang="en", model_type='facebook/bart-large', verbose=True, device=device)
        P, R, F1 = score(
            cands,
            refs,
            lang="en",
            model_type=model_type,
            verbose=False,
            device=device,
            use_fast_tokenizer=True,
            idf=True,
            batch_size=batch_size,
        )
        scores = {
            f"BERTScore-P ({model_type})": P,
            f"BERTScore-R ({model_type})": R,
            f"BERTScore-F1 ({model_type})": F1,
        }
    return scores


def evaluate_bert_score_multibatch(
    cands, refs, device="cuda", batch_size=32, model_type="bert-large-uncased"
):
    scores_list = []
    num_batch = len(cands) // batch_size
    final_scores = dict()
    for batch_i in tqdm.trange(num_batch + 1):
        cand_i = cands[batch_size * batch_i : batch_size * (batch_i + 1)]
        refs_i = refs[batch_size * batch_i : batch_size * (batch_i + 1)]
        if not cand_i:
            continue
        scores_list.append(evaluate_bert_score(cand_i, refs_i, device, model_type))

    final_scores[f"BERTScore-P ({model_type})"] = np.concatenate(
        [s[f"BERTScore-P ({model_type})"] for s in scores_list]
    )
    final_scores[f"BERTScore-R ({model_type})"] = np.concatenate(
        [s[f"BERTScore-R ({model_type})"] for s in scores_list]
    )
    final_scores[f"BERTScore-F1 ({model_type})"] = np.concatenate(
        [s[f"BERTScore-F1 ({model_type})"] for s in scores_list]
    )
    return final_scores
