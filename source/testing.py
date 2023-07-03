import torch
import os

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/.configs/" # workaround for pyplot config dir
import matplotlib.pyplot as plt

def get_model_predictions(batch_id, batch, sequences, args, mode="both"):
    pred_seq = output_sequneces_dict(sequences, args)
    gt_seq = get_gt_sequences(batch)

    stat_dict = dict()
    if mode in ["both", "seq"]:
        preds, target, indexes, stat_dict = get_sequence_tensors(pred_seq, gt_seq, batch_id, args)

    if mode in ["both", "feas"]:
        feasibility_y_hat, feasibility_y = get_feasibility_prediction(pred_seq, gt_seq)
        # feasilibility confusion matrix
        for k in ["Feas_TP", "Feas_TN", "Feas_FP", "Feas_FN", "Feas_P", "Feas_N"]:
            stat_dict[k] = 0

        if len(gt_seq):
            stat_dict["Feas_P"] = 1
        else:
            stat_dict["Feas_N"] = 1
        
        if feasibility_y:
            if feasibility_y_hat:
                stat_dict["Feas_TP"] = 1
            else:
                stat_dict["Feas_FN"] = 1
        else:
            if feasibility_y_hat:
                stat_dict["Feas_FP"] = 1
            else:
                stat_dict["Feas_TN"] = 1

    if mode == "both":
        return  preds, target, indexes, stat_dict, feasibility_y_hat, feasibility_y
    elif mode == "seq":
        return preds, target, indexes, stat_dict
    elif mode == "feas":
        return stat_dict, feasibility_y_hat, feasibility_y


def get_feasibility_prediction(pred_seq, gt_seq):
    y = 1 if len(gt_seq) else 0

    if len(pred_seq):
        # get the sequence with the maximal probability
        most_probable_seq = max(pred_seq.items(), key=lambda x: x[1]["total_prob"])[1]

        # get its minimal probablity along its sequnce
        y_hat = min(most_probable_seq["sequence_prob"])
    else:
        y_hat = 0

    return y_hat, y


def get_sequence_tensors(pred_seq, gt_seq, batch_id, args):

    union = pred_seq.keys() | gt_seq
    nof_seq = len(union)

    stat_dict = dict()

    stat_dict["Seq_TP"] = len(gt_seq & pred_seq.keys())
    stat_dict["Seq_FN"] = len(gt_seq - pred_seq.keys())
    stat_dict["Seq_FP"] = len(pred_seq.keys() - gt_seq)

    preds = torch.zeros(nof_seq, dtype=torch.float)
    target = torch.zeros(nof_seq, dtype=torch.long)
    indexes = torch.ones(nof_seq, dtype=torch.long) * batch_id
    
    pred_seqs = [list() for _ in range(nof_seq)]
    vars = [list() for _ in range(nof_seq)]

    for i, seq in enumerate(union):
        if seq in pred_seq:
            preds[i] = pred_seq[seq]["total_prob"]
            pred_seqs[i] = pred_seq[seq]["sequence_prob"]
            if args.mc_dropout:
                vars[i] = pred_seq[seq]["inter_var"]
        
        if seq in gt_seq:
            target[i] = 1

    stat_dict["sequence_prob"] = pred_seqs
    if args.mc_dropout:
        stat_dict["inter_var"] = vars
    
    return preds, target, indexes, stat_dict


def get_gt_sequences(batch):
    if isinstance(batch['part'].y, list):
        seq = batch['part'].y[0]
        if len(seq) == 1 and seq[0][0] == 'dummy':
            return set()
        return set([tuple(s) for s in seq])

    elif torch.is_tensor(batch['part'].y) and batch['part'].y.item() == 0:
        return set()

    raise ValueError("Something here is wrong!")


def output_sequneces_dict(sequences, args):
    out = dict()
    
    for seq in sequences:
        out_s = tuple()
        out_p = 1
        seq_p = list()
        inter_var = list()

        for item in seq:
            if isinstance(item['part'], int):
                out_s += (item['part'],)
                out_p *= item['prob']
                seq_p.append(item['prob'])
                if args.mc_dropout:
                    inter_var.append(item['inter_var'])
            elif isinstance(item['part'], str) and item['part'] == "fail":
                out_p = 0
                seq_p.append(0)

        out[out_s] = {"total_prob": out_p, "sequence_prob": seq_p}
        
        if args.mc_dropout:
            out[out_s]["inter_var"] = inter_var

    return out


def walk_tree(model, batch, args):
    out = list()

    # first, check if maybe we're already done
    if batch.x_dict['part'][:, 0].prod().item() == 1:
        seq = [{"part": "done", "prob": 1}]
        out.append(seq)
        return out

    # model output is part embeddings, need to call fc() to convert to logits
    if args.mc_dropout:
        logit = torch.stack([model.fc(model(batch)) for _ in range(args.mc_dropout)])
        probs = logit.sigmoid()
        var = probs.var(dim=0).squeeze(1).detach().numpy()
        probs = probs.mean(dim=0)
    else:
        logit = model.fc(model(batch))
        probs = logit.sigmoid()
    
    probs = probs.squeeze(1).detach().numpy()
    for i, p in enumerate(probs):
        if p <= args.threshold:
            continue

        # this node is alreayd assembled, shouldn't happen!
        if batch.x_dict['part'][i, 0].item() == 1:
            #print("Already assembed this one, skipping node..")
            continue

        # set node as assembled
        batch_copy = batch.detach().clone()
        batch_copy.x_dict['part'][i, 0] = 1

        # get possible sequences from this node onwards
        son_seq = walk_tree(model, batch_copy, args)

        # add this node to the head of the sequnces
        for s in son_seq:
            seq = [{"part": i, "prob": p}] + s if not args.mc_dropout else [{"part": i, "prob": p, "inter_var": var[i]}] + s 
            out.append(seq)

    # we've reached a deadend
    if len(out) == 0:
        seq = [{"part": "fail", "prob": 0}] if not args.mc_dropout else [{"part": "fail", "prob": 0, "inter_var": 0}]
        out.append(seq)

    return out

