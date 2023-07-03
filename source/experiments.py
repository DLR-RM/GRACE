from collections import defaultdict
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from os import path
import json
from tqdm import tqdm
import numpy as np
import sklearn.metrics as sk_metrics
from sklearn.model_selection import KFold
import random
import pickle
import torch_geometric.nn as pyg_nn

from source.data import GraphDataModule, GraphFoldDataModule
from source.model import AssemblyNodeLabelGNN, AssemblyNodeLabelWithDistanceGNN
from source.metrics import *
from source.utils import *
from source.testing import get_model_predictions, walk_tree, get_gt_sequences
from source.reporting import *
from source.dataset import read_filter_assemblies, create_sets, remove_files
from source.nf import NFModel


def train(config, args, device):

    config = set_missing_config_keys(config)
    config = update_data_dims(config)
    
    run = wandb.init(
        project="GNN4RoboAssembly",
        config=config,
        mode="online" if args.wandb else "disabled"
    )
    hparams = wandb.config

    print("User config:")
    print(json.dumps(config, sort_keys=True, indent=4))

    model = AssemblyNodeLabelWithDistanceGNN(hparams) if args.aux_loss else AssemblyNodeLabelGNN(hparams)
    model.to(device)

    logger = WandbLogger()
    logger.watch(model, log_graph=False)

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="val_acc", patience=10, verbose=True, mode="max"
    )
    dataset_name =  path.basename(path.dirname(args.dataset_path))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_acc', 
        mode="max", 
        dirpath=args.output_path, 
        filename='%s-{epoch:02d}-{val_acc:.2f}' % dataset_name
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=200,
        accelerator='auto',
        devices=1,
        logger=logger,
    )
    processed_dir_path = os.path.join(hparams.dataset_path, "processed/")
    print("Clearning up %s" % processed_dir_path)
    remove_files(processed_dir_path)

    data = GraphDataModule(hparams)

    with run:
        trainer.fit(model, data)


def step_by_step_test(args, device):

    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']

    # walkaround for old models
    hparams = set_missing_config_keys(hparams)

    model = AssemblyNodeLabelGNN(hparams)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.freeze()

    if args.mc_dropout:
        enable_dropout(model)

    hparams = model.hparams
    hparams.dataset_path = args.dataset_path
    data = GraphDataModule(hparams)

    dataloader = data.predict_dataloader(batch_size=1)

    parts_list = args.test_nof_parts if len(args.test_nof_parts) else [3, 4, 5, 6, 7] 

    probs = {n: torch.zeros(0, dtype=torch.long, device='cpu') for n in parts_list}
    gts = {n: torch.zeros(0, dtype=torch.long, device='cpu') for n in parts_list}

    for batch in tqdm(dataloader):
        n = batch['part'].x.shape[0]
        if n not in parts_list:
            continue

        # model output is part embeddings, need to call fc() to convert to logits
        if args.mc_dropout:
            logit = torch.stack([model.fc(model(batch)) for _ in range(args.mc_dropout)])
            prob = logit.sigmoid().mean(dim=0).detach().cpu()
        else:
            logit = model.fc(model(batch))
            prob = logit.sigmoid().detach().cpu()

        gt = batch['part'].y.detach().cpu()
        n = prob.shape[0]

        probs[n] = torch.cat([probs[n], prob.view(-1).cpu()])
        gts[n] = torch.cat([gts[n], gt.view(-1).cpu()])

    total_stats = defaultdict(int)
    for n in parts_list:
        print("############")
        print("PARTS: ", n)

        y_score = probs[n].numpy()
        y_pred = (probs[n] > args.threshold).float().numpy()
        y_true = gts[n].numpy()

        # tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred).ravel()
        stats = dict()
        stats["AUC"] = sk_metrics.roc_auc_score(y_true, y_score)
        stats["ACC"] = sk_metrics.accuracy_score(y_true, y_pred)
        stats["Precision"] = sk_metrics.precision_score(y_true, y_pred)
        stats["Recall"] = sk_metrics.recall_score(y_true, y_pred)

        sk_metrics.PrecisionRecallDisplay.from_predictions(y_true, y_score, name="%d parts" % n, ax=plt.gca())
        
        for metric in stats.keys():
            print(metric, ": ", stats[metric])
            total_stats[metric] += stats[metric]
        
    plt.title('Precision-Recall Curve')
    plt.savefig("step_by_steptest_curve.png")
    
    print("############")
    print("AVG Stats")
    for metric in total_stats.keys():
        print(metric, ": ", stats[metric]/len(parts_list))


def sequence_test(args, device):
    
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']

    print("User config:")
    print(json.dumps(hparams, sort_keys=True, indent=4))
    
    # walkaround for old models
    hparams = set_missing_config_keys(hparams)
    
    model = AssemblyNodeLabelWithDistanceGNN(hparams) if args.aux_loss else AssemblyNodeLabelGNN(hparams)
    model.load_state_dict(checkpoint["state_dict"])
    
    model.freeze() # set required_grad to false and model.eval()

    if args.mc_dropout:
        enable_dropout(model)

    hparams = model.hparams
    hparams.dataset_path = args.dataset_path
    data = GraphDataModule(hparams)

    #indices = [i for i in range(241)]
    #random.shuffle(indices)
    dataloader = data.predict_dataloader(batch_size=1,  indices=None)

    parts_list = args.test_nof_parts if len(args.test_nof_parts) else [3, 4, 5, 6, 7] 
    
    stats_funcs, feasibility_preds, feasibility_gt = dict(), dict(), dict()
    for n in parts_list:
        stats_funcs[n] = {
            #"Feasbile": {"func": SequenceStats(mode="feasbile"), "action": "box", "color": '#D7191C'},
            #"Infeasbile": {"func": SequenceStats(mode="infeasbile"), "action": "box", "color": '#2C7BB6'},
            "curve": {"func": SequencePrecisionRecallCurve(), "action": "curve"},
        }
        feasibility_preds[n] = list()
        feasibility_gt[n] = list()

    print("TESTING")
    confusion_stats = defaultdict(int)
    for batch_id, batch in enumerate(tqdm(dataloader)):
        n = batch['part'].x.shape[0]
        if n not in parts_list:
            continue
        
        sequences = walk_tree(model, batch, args)
        preds, target, indexes, stat_dict, feasibility_y_hat, feasibility_y =  get_model_predictions(batch_id, batch, sequences, args)

        for k in ["Seq_TP", "Seq_FN", "Seq_FP", "Feas_TP", "Feas_TN", "Feas_FP", "Feas_FN", "Feas_P", "Feas_N"]:
            confusion_stats[k] += stat_dict[k]

        for s in stats_funcs[n].keys():
            if hasattr(stats_funcs[n][s]["func"], "preds_seqs"):
                stats_funcs[n][s]["func"].update(preds, target, indexes, stat_dict)
            else:
                stats_funcs[n][s]["func"].update(preds, target, indexes)

        feasibility_preds[n].append(feasibility_y_hat)
        feasibility_gt[n].append(feasibility_y)

    print("Seqeunce Prediction: tp, fp, fn", [confusion_stats[k] for k in ["Seq_TP", "Seq_FP", "Seq_FN"]])
    print("Feasibility Prediction: tp, tn, fp, fn, p, n", [confusion_stats[k] for k in ["Feas_TP", "Feas_TN", "Feas_FP", "Feas_FN", "Feas_P", "Feas_N"]])

    print("REPORTING")
    #save_stat_boxplot(stats_funcs, parts_list, args)
    #save_histogram(stats_funcs, parts_list,)
    #print_stats(stats_funcs, parts_list)
    save_pr_curve(stats_funcs, parts_list)

    # feasiblity stats
    #save_roc_auc_curve(feasibility_preds, feasibility_gt)


def cross_validate(config, args, device):

    test_parts_list = args.test_nof_parts if len(args.test_nof_parts) else [3, 4, 5, 6, 7] 

    config = set_missing_config_keys(config)
    config = update_data_dims(config)

    print("User config:")
    print(json.dumps(config, sort_keys=True, indent=4))
    
    assemblies = read_filter_assemblies(args)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=32)

    mean_stats, seq_figs, seq_axis, step_figs, step_axis, pk_figs, pk_axis = dict(), dict(), dict(), dict(), dict(), dict(), dict()
    for n in test_parts_list:
        seq_figs[n], seq_axis[n]  = plt.subplots(1)
        step_figs[n], step_axis[n]  = plt.subplots(1)
        pk_figs[n], pk_axis[n]  = plt.subplots(1)
        mean_stats[n] = {
            "seq": {
                "pr": SequencePrecisionRecallCurve(), 
                "p@k": [list() for _ in range(10)],
                "auc": list()
            },
            "step": {
                "y_true": torch.zeros(0, dtype=torch.long, device='cpu'), 
                "y_score": torch.zeros(0, dtype=torch.long, device='cpu'), 
                "auc": list()
            }
        }
    
    for fold_id, (train_index, test_index) in enumerate(kf.split(assemblies)):
        print("#######################")
        print("CREATING FOLD %d" % fold_id)
        print("Assemblies Train/Test Split: ", len(train_index), len(test_index))

        trainval_set, test_set_seq, test_set_step, train_index, val_index = create_sets(assemblies, train_index, test_index, args)
        
        config["dataset_path"] = args.dataset_cache_root # hack of DataModule args
        run = wandb.init(
            project="GNN4RoboAssembly",
            config=config,
            mode="online" if args.wandb else "disabled",
        )
        hparams = wandb.config
        
        trainval_data = GraphFoldDataModule(hparams, trainval_set, "trainval", train_index, val_index, balanced_train_sampler=args.balanced_train_sampler)

        print("TRAINING FOLD %d" % fold_id)
        model = AssemblyNodeLabelGNN(hparams)
        model.to(device)
        model.train()

        logger = WandbLogger()
        logger.watch(model, log_graph=False)

        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="val_acc", patience=10, verbose=True, mode="max"
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_acc', 
            mode="max", 
            dirpath='/home/atad_ma/Projects/GNN4RoboAssembly/pl_checkpoints_cross_validation/', 
            filename='%d-%d-{epoch:02d}-{val_acc:.2f}' % (test_parts_list[0], fold_id)
        )

        trainer = pl.Trainer(
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=150,
            accelerator='auto',
            devices=1,
            logger=logger,
        )

        with run:
            trainer.fit(model, trainval_data)

        wandb.finish()

        if checkpoint_callback.best_model_score < args.skip_acc:
            print("Training failed, skipping fold..")
            for x in [model, trainval_set, test_set_seq, train_index, val_index, test_index]:
                x = None
            continue

        print("TESTING FOLD %d" % fold_id)
        model_path = checkpoint_callback.best_model_path
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        model.freeze() # set required_grad to false and model.eval()

        fold_stats = dict()
        for n in test_parts_list:
            fold_stats[n] = {
                "seq": {
                    "pr": SequencePrecisionRecallCurve(),
                    "p@k": [SequencePrecision(k=k) for k in range(1, 11)],
                }, 
                "step": {
                    "y_true": torch.zeros(0, dtype=torch.long, device='cpu'), 
                    "y_score": torch.zeros(0, dtype=torch.long, device='cpu')
                }
            }

        if args.test_mode in ["seq", "all"]:
            print("SEQUENCE TEST")
            test_data = GraphFoldDataModule(hparams, test_set_seq, "test")
            dataloader = test_data.predict_dataloader()

            # For test, threshold should be zero.
            # No worries, this won't effect next round training (fresh config is taken)
            args.threshold = 0

            for batch_id, batch in enumerate(tqdm(dataloader)):
                n = batch['part'].x.shape[0]
                
                sequences = walk_tree(model, batch, args)
                preds, target, indexes, stat_dict = get_model_predictions(batch_id, batch, sequences, args, mode="seq")

                fold_stats[n]["seq"]["pr"].update(preds, target, indexes, stat_dict)
                mean_stats[n]["seq"]["pr"].update(preds, target, indexes, stat_dict)
        
        if args.test_mode in ["p@k", "all"]:
            print("P@k TEST")
            test_data = GraphFoldDataModule(hparams, test_set_seq, "test")
            dataloader = test_data.predict_dataloader()

            args.threshold = 0.5

            for batch_id, batch in enumerate(tqdm(dataloader)):
                n = batch['part'].x.shape[0]
                
                sequences = walk_tree(model, batch, args)
                preds, target, indexes, stat_dict = get_model_predictions(batch_id, batch, sequences, args, mode="seq")
            
                for t in range(len(fold_stats[n]["seq"]["p@k"])):
                    fold_stats[n]["seq"]["p@k"][t].update(preds, target, indexes)
    
        
        if args.test_mode in ["step", "all"]:
            print("STEP-BY-STEP TEST")
            test_data = GraphFoldDataModule(hparams, test_set_step, "test")
            dataloader = test_data.predict_dataloader()

            for batch in tqdm(dataloader):
                # model output is part embeddings, need to call fc() to convert to logits
                if args.mc_dropout:
                    logit = torch.stack([model.fc(model(batch)) for _ in range(args.mc_dropout)])
                    prob = logit.sigmoid().mean(dim=0).detach().cpu()
                else:
                    logit = model.fc(model(batch))
                    prob = logit.sigmoid().detach().cpu()

                gt = batch['part'].y.detach().cpu()
                n = prob.shape[0]

                fold_stats[n]["step"]["y_true"] = torch.cat([fold_stats[n]["step"]["y_true"], gt.view(-1).cpu()])
                mean_stats[n]["step"]["y_true"] = torch.cat([mean_stats[n]["step"]["y_true"], gt.view(-1).cpu()])
                fold_stats[n]["step"]["y_score"] = torch.cat([fold_stats[n]["step"]["y_score"], prob.view(-1).cpu()])
                mean_stats[n]["step"]["y_score"] = torch.cat([mean_stats[n]["step"]["y_score"], prob.view(-1).cpu()])


        print("RESULTS FOLD %d" % fold_id)
        for n in test_parts_list:
            print("%d parts:" % n)
            # SEQ STATS
            if args.test_mode in ["seq", "all"]:
                ps, rs, _ = fold_stats[n]["seq"]["pr"].compute()
                disp = sk_metrics.PrecisionRecallDisplay(ps, rs)
                auc = sk_metrics.auc(rs, ps)

                mean_stats[n]["seq"]["auc"].append(auc)
                actual_fold_id = len(mean_stats[n]["seq"]["auc"])

                label = "fold-%d (AUC=%.2f)" % (actual_fold_id, auc)
                disp.plot(ax=seq_axis[n], name=label, lw=1, alpha=0.3)
                print("Seq:" + label)

                seq_axis[n].set_title("Complete Seqeunce Precision-Recall, %d parts" % n)
                seq_axis[n].legend()
                seq_axis[n].set_xlim([-0.05, 1.05])
                seq_axis[n].set_ylim([-0.05, 1.05])

                seq_figs[n].savefig("%d_seq_pr_curve.png" % (n))

            # P@K STATS
            if args.test_mode in ["p@k", "all"]:
                pks = list()
                for t in range(len(fold_stats[n]["seq"]["p@k"])):
                    pk = fold_stats[n]["seq"]["p@k"][t].compute().item()
                    
                    # confirm p@k make sense for this k
                    if pk != -1:
                        pks.append(pk)
                        mean_stats[n]["seq"]["p@k"][t].append(pk)
                
                print("P@k: ", pks)
                actual_fold_id = len(mean_stats[n]["seq"]["p@k"][0])
            
            # STEP STATS
            if args.test_mode in ["step", "all"]:
                y_true = fold_stats[n]["step"]["y_true"].numpy()
                y_score = fold_stats[n]["step"]["y_score"].numpy()

                ps, rs, _ = sk_metrics.precision_recall_curve(y_true, y_score)
                disp = sk_metrics.PrecisionRecallDisplay(ps, rs)
                auc = sk_metrics.auc(rs, ps)
                
                mean_stats[n]["step"]["auc"].append(auc)
                actual_fold_id = len(mean_stats[n]["step"]["auc"])
                
                label = "fold-%d (AUC=%.2f)" % (actual_fold_id, auc)
                disp.plot(ax=step_axis[n], name=label, lw=1, alpha=0.3)
                print("Step:" +label)

                step_axis[n].set_title("Step-by-Step Precision Recall, %d parts" % n)
                step_axis[n].legend()
                step_axis[n].set_xlim([-0.05, 1.05])
                step_axis[n].set_ylim([-0.05, 1.05])

                step_figs[n].savefig("%d_step_pr_curve.png" % (n))

        for x in [model, trainval_set, test_set_seq, test_set_step, train_index, val_index, test_index, fold_stats]:
            x = None

        if actual_fold_id == 4:
            print("We have 4 suceesfull folds, done..")
            break

    print("FINAL RESULTS")
    for n in test_parts_list:
        print("%d parts:" % n)
        if args.test_mode in ["seq", "all"]:
            ps, rs, _ = mean_stats[n]["seq"]["pr"].compute()
            disp = sk_metrics.PrecisionRecallDisplay(ps, rs)

            mean_auc = np.mean(mean_stats[n]["seq"]["auc"])
            std_auc = np.std(mean_stats[n]["seq"]["auc"])

            label = "Mean (AUC=%.2f±%.2f)" % (mean_auc, std_auc)
            disp.plot(ax=seq_axis[n], name=label, lw=2, alpha=.8, color='b')
            print("Seq: " +  label)

            seq_axis[n].set_title("Complete Seqeunce Precision-Recall, %d parts" % n)
            seq_axis[n].legend()
            seq_axis[n].set_xlim([-0.05, 1.05])
            seq_axis[n].set_ylim([-0.05, 1.05])

            seq_figs[n].savefig("%d_seq_pr_curve.png" % (n))
        
        if args.test_mode in ["p@k", "all"]:
            for t in range(len(mean_stats[n]["seq"]["p@k"])):
                pks = mean_stats[n]["seq"]["p@k"][t]
                if not len(pks):
                    continue
                
                k = t + 1

                pk_axis[n].scatter([k for _ in range(len(pks))], pks, alpha=.3)
                mean = np.mean(pks)
                pk_axis[n].scatter(k, mean, alpha=.8, color='b', marker='x')

                std = np.std(pks)
                print("P@%d: %.2f±%.2f" % (k, mean, std))

            pk_axis[n].set_title("P@k, %d parts" % n)
            pk_axis[n].set_xlim([0, 11])
            pk_axis[n].set_xticks(range(1,11))
            pk_axis[n].set_ylim([0, 1])
            pk_axis[n].set_xlabel("k")
            pk_axis[n].set_ylabel("Precision@k")

            pk_figs[n].savefig("%d_pk.png" % (n))

        if args.test_mode in ["step", "all"]:
            y_true = mean_stats[n]["step"]["y_true"].numpy()
            y_score = mean_stats[n]["step"]["y_score"].numpy()

            ps, rs, _ = sk_metrics.precision_recall_curve(y_true, y_score)
            disp = sk_metrics.PrecisionRecallDisplay(ps, rs)

            mean_auc = np.mean(mean_stats[n]["step"]["auc"])
            std_auc = np.std(mean_stats[n]["step"]["auc"])

            label = "Mean (AUC=%.2f±%.2f)" % (mean_auc, std_auc)
            disp.plot(ax=step_axis[n], name=label, lw=2, alpha=.8, color='b')

            print("Step:" +label)

            step_axis[n].set_title("Step-by-Step Precision Recall, %d parts" % n)
            step_axis[n].legend()
            step_axis[n].set_xlim([-0.05, 1.05])
            step_axis[n].set_ylim([-0.05, 1.05])

            step_figs[n].savefig("%d_step_pr_curve.png" % (n))

'''
def aux_loss_test(args, device):

    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']

    model = AssemblyNodeLabelWithDistanceGNN(hparams)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    
    model.freeze() # set required_grad to false and model.eval()

    hparams = model.hparams
    hparams.dataset_path = args.dataset_path
    data = GraphDataModule(hparams)

    dataloader = data.predict_dataloader(val_set=True, batch_size=1)

    parts_list = [3, 4, 5, 6, 7]

    preds = {n: torch.zeros(0, dtype=torch.long, device='cpu') for n in parts_list}
    gts = {n: torch.zeros(0, dtype=torch.long, device='cpu') for n in parts_list}

    for batch in tqdm(dataloader):
        n = batch['part'].x.shape[0]

        _, part_pos = model(batch)

        src, dst = batch.edge_index_dict[('part', 'dist', 'part')]
        pred = torch.norm(part_pos[src] - part_pos[dst], p=2, dim=-1)

        gt = batch.edge_weight_dict[('part', 'dist', 'part')]

        preds[n] = torch.cat([preds[n], pred.view(-1).cpu()])
        gts[n] = torch.cat([gts[n], gt.view(-1).cpu()])

    for n in parts_list:
        plt.clf()

        y_score = preds[n].numpy()
        y_true = gts[n].numpy()
        
        plt.hist(y_score, alpha=0.5, label="%d, Pred" % n, bins=100)
        plt.hist(y_true, alpha=0.5, label="%d, GT" % n, bins=100)
        
        plt.legend()
        plt.title("Distances Distribution")
        plt.savefig("hist_%d.png" % n)
'''


def train_nf(args, device):
    # load GNN model
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    gnn_hparams = checkpoint['hyper_parameters']

    # walkaround for old models
    gnn_hparams = set_missing_config_keys(gnn_hparams)

    if "part_dist_fc.weight" in checkpoint["state_dict"]:
        gnn_model = AssemblyNodeLabelWithDistanceGNN(gnn_hparams) 
    else:
        gnn_model = AssemblyNodeLabelGNN(gnn_hparams)
    gnn_model.load_state_dict(checkpoint["state_dict"])

    # load NF model
    hparams = {
        'flow_layers': args.nf_layers, 
        'learning_rate': args.nf_lr, 
        'batch_size': args.nf_batch,
        'input_dim': 94,
        'nf_layer_size': 94,
        'nf_scale_map': "exp",
        'nf_base_dist': args.nf_base_dist,
        'nf_supervised_loss': args.nf_supervised_loss,
        'nf_c_threshold': args.nf_c_threshold,
        'nf_act_norm': args.nf_act_norm,
    }
    
    wandb.init(
        project="GNN4RoboAssembly NF",
        config=hparams,
        mode="offline"
    )
    hparams = wandb.config
    nf_model = NFModel(gnn_model, hparams)

    # load dataset, make sure same tranformations are used as the ones used for training the GNN
    data_hparams = gnn_hparams
    data_hparams['learning_rate'] = hparams['learning_rate']
    data_hparams['batch_size'] = hparams['batch_size']
    
    data_hparams['dataset_path'] = args.dataset_path
    train_data = GraphDataModule(data_hparams)

    if args.nf_val_set:
        data_hparams['dataset_path'] = args.nf_val_set
        val_data = GraphDataModule(data_hparams)
    
    logger = WandbLogger(offline=True)
    logger.watch(nf_model, log_graph=False)

    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="nf_val_auc", patience=10, verbose=True, mode="max", check_finite=True
    )
    dataset_name = path.basename(path.dirname(args.dataset_path))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='nf_val_auc', 
        mode="max", 
        dirpath=args.output_path, 
        filename='nf_baseline_%s-{epoch:02d}-{nf_val_auc:.2f}' % dataset_name,
    )

    trainer = pl.Trainer(
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=500,
        accelerator='auto',
        devices=1,
        logger=logger,
    )

    if not args.nf_val_set:
        trainer.fit(nf_model, train_data)
    else:
        train_loader = train_data.predict_dataloader(batch_size=hparams['batch_size'])
        val_loader = val_data.predict_dataloader(batch_size=hparams['batch_size'])
        trainer.fit(nf_model, train_loader, val_loader)
        

def nf_classifier(args, device):
    # load GNN model
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    gnn_hparams = checkpoint['hyper_parameters']

    # walkaround for old models
    gnn_hparams = set_missing_config_keys(gnn_hparams)

    if "part_dist_fc.weight" in checkpoint["state_dict"]:
        gnn_model = AssemblyNodeLabelWithDistanceGNN(gnn_hparams) 
    else:
        gnn_model = AssemblyNodeLabelGNN(gnn_hparams)
    gnn_model.load_state_dict(checkpoint["state_dict"])

    # load NF model
    checkpoint = torch.load(args.nf_ckpt, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']
    if 'input_dim' not in hparams:
        hparams['input_dim'] = 94
    
    run = wandb.init(
        project="GNN4RoboAssembly NF",
        config=hparams,
        mode="offline"
    )
    hparams = wandb.config
    nf_model = NFModel(gnn_model, hparams)
    nf_model.load_state_dict(checkpoint["state_dict"])
    nf_model.freeze()

    # load dataset, make sure same tranformations are used as the ones used for training the GNN
    data_hparams = gnn_hparams
    data_hparams['learning_rate'] = hparams['learning_rate']
    data_hparams['batch_size'] = hparams['batch_size']

    y_score, y_true, feasible, infeasible = list(), list(), list(), list()

    for file in ["../datasets/gseq_5_nf_test/"]:
        # which pooling, which function, size of layers
        data_hparams['dataset_path'] = file
        data = GraphDataModule(data_hparams)

        dataloader = data.predict_dataloader(batch_size=1,  indices=None)

        for _, batch in enumerate(tqdm(dataloader)):
            pred = nf_model(batch)
            pred = pred.detach().cpu().item()

            gt_seq = get_gt_sequences(batch)
            y = 1 if len(gt_seq) else 0

            if y:
                feasible.append(pred)
            else:
                infeasible.append(pred)

            y_score.append(pred)
            y_true.append(y)
    
    feasible = np.array(feasible)
    infeasible = np.array(infeasible)

    print("Feasible mean, std", feasible.mean(), feasible.std())
    print("Infeasible mean, std", infeasible.mean(), infeasible.std())

    y_true = np.array(y_true)
    y_score = np.nan_to_num(np.array(y_score))

    sk_metrics.RocCurveDisplay.from_predictions(y_true, y_score)

    plt.title('RoC Curve')
    plt.xlim((0, 1))
    plt.savefig("roc_curve_feasibility.png")


def compare_feasibility_cls(args, device):

    # set classifiers checkpoints here, specify their types

    output_path = args.output_path
    classifiers = [
        {
            "ckpt": "pl_checkpoints/nml_5_trainval-epoch=32-val_acc=0.99.ckpt",
            "nf_ckpt": "pl_nf_checkpoints/nf_baseline_gseq_5_nf_train-epoch=27-nf_val_auc=0.85.ckpt",
            "name": "NF",
            "type": "nf",
        }
    ]

    '''
    classifiers = [
        {
            "ckpt": "experiments/feasiblity/nml_5_trainval_mixed-epoch=37-val_acc=0.98.ckpt",
            "name": "# sequences",
            "type": "sequence",
        },
        {
            "ckpt": "experiments/feasiblity/nml_5_trainval_mixed-epoch=37-val_acc=0.98.ckpt",
            "name": "SVM (Linear)",
            "estimator": "benchmark_models/2class/Linear SVM.pkl",
            "scaler": "benchmark_models/2class/5parts_scaler.pkl",
            "type": "benchmark",
        },
        {
            "ckpt": "experiments/feasiblity/nml_5_trainval_mixed-epoch=37-val_acc=0.98.ckpt",
            "name": "SVM (RBF)",
            "estimator": "benchmark_models/2class/RBF SVM.pkl",
            "scaler": "benchmark_models/2class/5parts_scaler.pkl",
            "type": "benchmark",
        },
        {
            "ckpt": "experiments/feasiblity/nml_5_trainval_mixed-epoch=37-val_acc=0.98.ckpt",
            "name": "NN",
            "estimator": "benchmark_models/2class/Neural Net.pkl",
            "scaler": "benchmark_models/2class/5parts_scaler.pkl",
            "type": "benchmark",
        },
        {
            "ckpt": "experiments/feasiblity/nml_5_trainval_mixed-epoch=37-val_acc=0.98.ckpt",
            "name": "Nearest Neighbor",
            "estimator": "benchmark_models/2class/Nearest Neighbors.pkl",
            "scaler": "benchmark_models/2class/5parts_scaler.pkl",
            "type": "benchmark",
        },
        {
            "ckpt": "experiments/feasiblity/nml_5_trainval_feasible-epoch=22-val_acc=0.98.ckpt",
            "name": "# sequences (feasible only)",
            "type": "sequence",
        }, 
    ]
    '''

    '''
    # compare classifiers, 5 parts setting
    classifiers = [
        {
            "ckpt": "pl_checkpoints/nml_all-epoch=33-val_acc=0.99.ckpt",
            "name": "Sequence Set Size, 1-class",
            "type": "sequence",
        }, 
        {
            "ckpt": "experiments/feasiblity/nml_5_trainval_mixed-epoch=37-val_acc=0.98.ckpt",
            "name": "Sequence Set Size, 2-class",
            "type": "sequence",
        },
        {
            "ckpt": "pl_checkpoints/nml_5_trainval-epoch=32-val_acc=0.99.ckpt",
            "nf_ckpt": "pl_nf_checkpoints/longer_gseq_5_nf_train-epoch=32-nf_val_auc=0.85.ckpt",
            "name": "NF, 1-class",
            "type": "nf",
        },
        {
            "ckpt": "pl_checkpoints/nml_5_trainval-epoch=32-val_acc=0.99.ckpt",
            "name": "SVM, 1-class",
            "estimator": "benchmark_models/5parts_svm.pkl",
            "scaler": "benchmark_models/5parts_scaler.pkl",
            "type": "benchmark",
        },
    ]
    
    # compare classifiers, 6 parts setting
    classifiers = [
        {
            "ckpt": "pl_checkpoints/nml_all-epoch=33-val_acc=0.99.ckpt",
            "name": "Sequence Set Size, 1-class",
            "type": "sequence",
        }, 
        {
            "ckpt": "pl_checkpoints/nml_with_infeasible_inter-epoch=105-val_loss=0.00-lr=0.001724014701500319.ckpt",
            "name": "Sequence Set Size, 2-class",
            "type": "sequence",
        },
        {
            "ckpt": "pl_checkpoints/nml_6_trainval-epoch=57-val_acc=0.99.ckpt",
            "nf_ckpt": "pl_nf_checkpoints/6_feasbility_experiment-epoch=27-nf_val_auc=0.80.ckpt",
            "name": "NF, 1-class",
            "type": "nf",
        },
        {
            "ckpt": "pl_checkpoints/nml_6_trainval-epoch=57-val_acc=0.99.ckpt",
            "name": "SVM, 1-class",
            "estimator": "benchmark_models/6parts_svm.pkl",
            "type": "benchmark",
        },
    ]
    '''
    

    # set dataset paths
    datasets = {
        "mixed": "../datasets/feasbility_experiment/gseq_5_test_mixed/",
        "feasible": "../datasets/feasbility_experiment/gseq_test_feasible_only/"
    }

    # set required tests
    tests = {
        "seq": False, "feas": True
    }

    # just make sure
    args.threshold = 0
    parts_list = args.test_nof_parts if len(args.test_nof_parts) else [5]
    print("Testing on %s parts only!" % parts_list)

    # init figures
    seq_figs, seq_axis, feas_figs, feas_axis = dict(), dict(), dict(), dict()
    for n in parts_list:
        seq_figs[n], seq_axis[n]  = plt.subplots(1)
        feas_figs[n], feas_axis[n]  = plt.subplots(1)

    for i, cls in enumerate(classifiers):
        print("Classifer %d" % i)
        print(cls)

        class_stats = dict()
        for n in parts_list:
            class_stats[n] = {
                "seq": {
                    "pr": SequencePrecisionRecallCurve(),
                }, 
                "feas": {
                    "y_true": list(), "y_score": list()
                }
            }
        
        checkpoint = torch.load(cls['ckpt'], map_location=torch.device('cpu'))
        hparams = checkpoint['hyper_parameters']
        
        # walkaround for old models
        hparams = set_missing_config_keys(hparams)
        
        model = AssemblyNodeLabelWithDistanceGNN(hparams) if args.aux_loss else AssemblyNodeLabelGNN(hparams)
        model.load_state_dict(checkpoint["state_dict"])
        model.freeze() # set required_grad to false and model.eval()

        hparams = model.hparams

        if args.mc_dropout:
            enable_dropout(model)

        if cls["type"] == "nf":
            checkpoint = torch.load(cls['nf_ckpt'], map_location=torch.device('cpu'))
            nf_hparams = checkpoint['hyper_parameters']

            nf_model = NFModel(model, nf_hparams)
            nf_model.load_state_dict(checkpoint["state_dict"])
            nf_model.freeze()

            # over-write GNN params w/ NF params for dataloader
            hparams['learning_rate'] = nf_hparams['learning_rate']
            hparams['batch_size'] = nf_hparams['batch_size']
        
        elif cls["type"] == "benchmark":
            estimator = pickle.load(open(cls['estimator'], 'rb'))
            scaler = pickle.load(open(cls['scaler'], 'rb')) if 'scaler' in cls else None

        # feasbility prediction test
        if tests["feas"]:
            hparams.dataset_path = datasets["mixed"] 
            data = GraphDataModule(hparams)
            dataloader = data.predict_dataloader(batch_size=1,  indices=None)
            
            for batch_id, batch in enumerate(tqdm(dataloader)):
                n = batch['part'].x.shape[0]
                if n not in parts_list:
                    continue
                
                if cls["type"] in ["nf", "benchmark"]:
                    if cls["type"] == "nf":
                        y_score = nf_model(batch)
                        y_score = y_score.detach().cpu().item()
                    else:
                        part_embed = model(batch)
                        graph_embed = pyg_nn.global_mean_pool(part_embed, batch=batch['part'].batch)
                        graph_embed = graph_embed.cpu().detach().numpy()
                        graph_embed = scaler.transform(graph_embed) if scaler else graph_embed
                        y_score = estimator.decision_function(graph_embed) if hasattr(estimator, "decision_function") else estimator.predict_proba(graph_embed)[:, 1]

                    gt_seq = get_gt_sequences(batch)
                    y_true = 1 if len(gt_seq) else 0

                elif cls["type"] == "sequence":
                    sequences = walk_tree(model, batch, args)
                    _, y_score, y_true =  get_model_predictions(batch_id, batch, sequences, args, mode="feas")

                class_stats[n]["feas"]["y_score"].append(y_score)
                class_stats[n]["feas"]["y_true"].append(y_true)

        # seqeunce prediction test
        if tests["seq"]:
            hparams.dataset_path = datasets["feasible"]
            data = GraphDataModule(hparams)
            dataloader = data.predict_dataloader(batch_size=1,  indices=None)
            
            for batch_id, batch in enumerate(tqdm(dataloader)):
                n = batch['part'].x.shape[0]
                if n not in parts_list:
                    continue
                
                sequences = walk_tree(model, batch, args)
                preds, target, indexes, stat_dict =  get_model_predictions(batch_id, batch, sequences, args, mode="seq")

                class_stats[n]["seq"]["pr"].update(preds, target, indexes, stat_dict)

        # reporting
        for n in parts_list:
            # sequence PR curve
            if tests["seq"]:
                ps, rs, _ = class_stats[n]["seq"]["pr"].compute()
                disp = sk_metrics.PrecisionRecallDisplay(ps, rs)
                auc = sk_metrics.auc(rs, ps)
                label = "%s (AUC=%.2f)" % (cls["name"], auc)
                disp.plot(ax=seq_axis[n], name=label)
                seq_figs[n].savefig(path.join(output_path, "%d_seq_pr_curve.pdf" % (n)))

            # feasibility roc auc
            if tests["feas"]:
                y_true = np.array(class_stats[n]["feas"]["y_true"])
                y_pred = np.array(class_stats[n]["feas"]["y_score"])
                y_pred = np.nan_to_num(y_pred)
                sk_metrics.RocCurveDisplay.from_predictions(y_true, y_pred, ax=feas_axis[n], name=cls["name"])
                feas_figs[n].savefig(path.join(output_path, "%d_feas_roc_curve.pdf" % (n)))
    
    # final prepartion of figures
    for n in parts_list:
        if tests["seq"]:
            seq_axis[n].axline((1, 0), slope=0, color="navy", lw=2, linestyle="--", label="chance")
            seq_axis[n].set_title("Seqeunce Prediction Precision-Recall, %d parts" % n)
            seq_axis[n].legend()
            seq_axis[n].set_xlim([-0.05, 1.05])
            seq_axis[n].set_ylim([-0.05, 1.05])
            seq_figs[n].savefig(path.join(output_path, "%d_seq_pr_curve.pdf" % (n)))

        if tests["feas"]:
            feas_axis[n].axline((1, 1), slope=1, color="navy", lw=2, linestyle="--", label="chance")
            feas_axis[n].set_title("Feasibility Prediction ROC, %d parts" % n)
            feas_axis[n].legend(loc='lower right')
            feas_axis[n].set_xlim([-0.05, 1.05])
            feas_axis[n].set_ylim([-0.05, 1.05])
            feas_figs[n].savefig(path.join(output_path, "%d_feas_roc_curve.pdf" % (n)))

    print("Done!")


def ablation_study(config, args, device):
    
    parts_list = args.test_nof_parts if len(args.test_nof_parts) else [3, 4, 5, 6, 7]

    datasets = {
        "trainval": "../datasets/feasbility_experiment/nml_trainval_feasible_only/",
        "test": "../datasets/feasbility_experiment/gseq_test_feasible_only/"
    }

    tests = {      
        "GAT backbone": {
            "ckpt": "experiments/ablations/gat_0-epoch=28-val_acc=0.99.ckpt",
            "train_config": {"architecture": "GAT"}, 
            "test_config": dict(), 
        }, 
        "GCN backbone": {
            "ckpt": "experiments/ablations/gcn_0-epoch=46-val_acc=1.00.ckpt",
            "train_config": {"architecture": "GCN"}, 
            "test_config": dict(), 
        },        
    }


    '''
    tests = {       
        "Baseline": {
            "ckpt": "experiments/ablations/self_loops_nml_trainval_feasible_only-epoch=63-val_acc=0.98.ckpt",
            "train_config": dict(), 
            "test_config": dict(), 
        },
        "Random values": {
            "ckpt": "experiments/ablations/random_pos_encoding-epoch=14-val_acc=0.94.ckpt",
            "train_config": {"pos_encoding": False, "random_pos_encoding": True},
            "test_config": dict(), 
        },
        "No encoding": {
            "ckpt": "experiments/ablations/no_pos_encoding-epoch=26-val_acc=0.88.ckpt",
            "train_config": {"pos_encoding": False, "remove_pos_encodings": True,"part_input_dim": 2, "surface_input_dim": 1},
            "test_config": dict(), 
        }
    }
    
    tests = {       
        "Baseline": {
            "ckpt": "experiments/ablations/self_loops_nml_trainval_feasible_only-epoch=63-val_acc=0.98.ckpt",
            "train_config": dict(), 
            "test_config": dict(), 
        },
        "Part permutation (Test)": {
            "ckpt": "experiments/ablations/self_loops_nml_trainval_feasible_only-epoch=63-val_acc=0.98.ckpt",
            "train_config": dict(),
            "test_config": {"permute_parts": True},
        },
        "Surface permutation (Test)": {
            "ckpt": "experiments/ablations/self_loops_nml_trainval_feasible_only-epoch=63-val_acc=0.98.ckpt",
            "train_config": dict(),
            "test_config": {"permute_surfaces": True},
        },
        "Part permutation (Train+Test)": {
            "ckpt": "experiments/ablations/part_permut_3-epoch=37-val_acc=0.99.ckpt",
            "train_config": {"permute_parts": True},
            "test_config": {"permute_parts": True},
        },
        "Surface permutation (Train+Test)": {
            "ckpt": "experiments/ablations/surface_permut_4-epoch=53-val_acc=0.89.ckpt",
            "train_config": {"permute_surfaces": True},
            "test_config": {"permute_surfaces": True},
        },
    }
    '''

    # init figures
    seq_figs, seq_axis = dict(), dict()
    for n in parts_list:
        seq_figs[n], seq_axis[n]  = plt.subplots(1)

    for test_id, (test_name, test) in enumerate(tests.items()):
        print("Test: %s" % test_name)

        if test["ckpt"] is None or len(test["ckpt"]) == 0:
            # we may change pre-processing in this test, need to remove processed folder
            processed_dir_path = os.path.join(datasets["trainval"], "processed/")
            print("Clearning up %s" % processed_dir_path)
            remove_files(processed_dir_path)

            train_config = config.copy()
            train_config["dataset_path"] = datasets["trainval"]
             # update default config per specificaiton
            changes = test["train_config"]
            if len(changes):
                for key, new_value in changes.items():
                    train_config[key] = new_value

            train_config = set_missing_config_keys(train_config)
            train_config = update_data_dims(train_config)

            print("Train config:")
            print(json.dumps(train_config, sort_keys=True, indent=4))

            wandb.init(
                project="GNN4RoboAssembly",
                config=train_config,
                mode="disabled",
                allow_val_change=True
            )
            hparams = wandb.config

            model = AssemblyNodeLabelGNN(hparams)
            model.to(device)

            early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_acc", patience=10, verbose=True, mode="max"
            )
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor='val_acc', 
                mode="max", 
                dirpath='/home/atad_ma/Projects/GNN4RoboAssembly/experiments/ablations/', 
                filename='%d-{epoch:02d}-{val_acc:.2f}' % test_id
            )

            trainer = pl.Trainer(
                callbacks=[checkpoint_callback, early_stop_callback],
                max_epochs=200,
                accelerator='auto',
                devices=1,
            )
            data = GraphDataModule(hparams)
            trainer.fit(model, data)
            test["ckpt"] = checkpoint_callback.best_model_path

        # we may change pre-processing in this test, need to remove processed folder
        processed_dir_path = os.path.join(datasets["test"], "processed/")
        print("Clearning up %s" % processed_dir_path)
        remove_files(processed_dir_path)    

        checkpoint = torch.load(test["ckpt"], map_location=torch.device('cpu'))
        hparams = checkpoint['hyper_parameters']
        hparams["dataset_path"] = datasets["test"]
        hparams = set_missing_config_keys(hparams)

        # update test config per specificaiton
        changes = test["test_config"] if "test_config" in test else []
        if len(changes):
            for key, new_value in changes.items():
                hparams[key] = new_value

        print("Test config:")
        print(json.dumps(hparams, sort_keys=True, indent=4))

        model = AssemblyNodeLabelGNN(hparams)
        model.load_state_dict(checkpoint["state_dict"])
        model.freeze() # set required_grad to false and model.eval()

        data = GraphDataModule(hparams)
        dataloader = data.predict_dataloader(batch_size=1,  indices=None)

        # testing
        stats = { n: SequencePrecisionRecallCurve() for n in parts_list}
        args.threshold = 0
        for batch_id, batch in enumerate(tqdm(dataloader)):
            n = batch['part'].x.shape[0]
            if n not in parts_list:
                continue
            
            sequences = walk_tree(model, batch, args)
            preds, target, indexes, stat_dict =  get_model_predictions(batch_id, batch, sequences, args, mode="seq")

            stats[n].update(preds, target, indexes, stat_dict)

         # reporting
        for n in parts_list:
            ps, rs, _ = stats[n].compute()
            disp = sk_metrics.PrecisionRecallDisplay(ps, rs)
            auc = sk_metrics.auc(rs, ps)
            label = "%s (AUC=%.2f)" % (test_name, auc)
            disp.plot(ax=seq_axis[n], name=label)
            seq_figs[n].savefig("%d_seq_pr_curve.png" % (n))

    
    # final prepartion of figures
    for n in parts_list:
        seq_axis[n].axline((1, 0), slope=0, color="navy", lw=2, linestyle="--", label="chance")
        seq_axis[n].set_title("Seqeunce Prediction Precision-Recall, %d parts" % n)
        seq_axis[n].legend(loc=(1.05, 0.5))
        seq_axis[n].set_xlim([-0.05, 1.05])
        seq_axis[n].set_ylim([-0.05, 1.05])

        seq_figs[n].savefig("%d_seq_pr_curve.png" % (n), bbox_inches="tight")

    print("Done!")
