import argparse
import pytorch_lightning as pl

from source.experiments import *
from source.utils import update_data_dims

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    pl.seed_everything(32, workers=True)
    np.set_printoptions(precision=2, suppress=True)

    # model training conifg, testing config is taken from checkpoint
    config = {
        # dataset
        "train_ratio": args.train_ratio,
        "dataset_path": args.dataset_path,
        "part_input_dim": 3,
        "surface_input_dim": 2,
        "edge_dim": 1,
        "pos_encoding": True,
        "encoding_length": 16,
        "one_hot": False,
        "edge_normalization": False,
        "num_workers": 8,
        "permute_nodes": False,
        # training
        "batch_size": 256,
        "learning_rate": args.learning_rate, # 0.002182, 0.001724
        "reg_delta": args.reg_delta,
        "optimizer": "Adam",
        "loss_function": "BCE",
        "threshold": args.threshold, # 0.87
        # model
        "architecture": "GAT",
        "hidden_channels": 94,
        "surface_gnn_layers": 3,
        "part_gnn_layers": 1,
        "fc_layers": 1,
        "relu_slope": 0.15,
        "instance_norm_affine": True,
        "instance_norm_running_stats": False,
        "dropout": args.dropout,
        # aux loss
        "part_dist_fc_channels": 17,
        "aux_loss_delta": 0.4
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train(config, args, device)
    
    elif args.mode == "step_by_step_test":
        step_by_step_test(args, device)

    elif args.mode == "sequence_test":
        sequence_test(args, device)
    
    elif args.mode == "cross_validate":
        cross_validate(config, args, device)
    
    elif args.mode == "nf":
        train_nf(args, device)
    
    elif args.mode == "nf_classifier":
        nf_classifier(args, device)

    elif args.mode == "compare":
        compare_feasibility_cls(args, device)

    elif args.mode == "ablation_study":
        ablation_study(config, args, device)

    else:
        raise ValueError("Unrecognized operation mode %s" % args.mode)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dataset_path", type=str, default="/home/atad_ma/Projects/datasets/")
    parser.add_argument("--output_path", type=str, default="/home/atad_ma/Projects/GNN4RoboAssembly/pl_checkpoints")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Train set size")
    parser.add_argument("--learning_rate", type=float, default=0.002182, help="Learning Rate")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout Rate")
    parser.add_argument("--reg_delta", type=float, default=0.3, help="Regularization term delta")

    parser.add_argument("--wandb", action='store_true')

    parser.add_argument("--mode", type=str, default="step_by_step_test")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file name to use for test", 
                        default="nml_all-epoch=33-val_acc=0.99.ckpt")
    
    parser.add_argument("--threshold", type=float, default=0.87, help="Threshold for step-by-step")
    parser.add_argument("--mc_dropout", type=int, default=0, help="Monte-Carlo Test Dropout forward passes, 0 for disabled")

    # aux loss
    parser.add_argument("--aux_loss",  action='store_true', help="Should distance be used as aux loss")

    # cross validate args
    parser.add_argument("--raw_dir", type=str, default="/home/rodr_is/data_matan/", help="Path to assembly folders")
    parser.add_argument("--assemblies_cache", type=str, default="/home/atad_ma/Projects/datasets/all_cache.pkl", help="Cache of preprocessed raw files")
    parser.add_argument("--dataset_cache_root", type=str, default="/home/atad_ma/Projects/datasets/cache/", help="Cache of preprocessed raw files")
    parser.add_argument("--infeasible", type=str, default="exclude_all", help="Either exclude_all, intermediate_only, include_all")
    parser.add_argument("--ranges", type=str, nargs='*', default="0,19999", help="Index range for assembly directory")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--train_nof_parts", nargs="*", type=int, default=[], help="Train number of parts")
    parser.add_argument("--val_nof_parts", nargs="*", type=int, default=[], help="Val number of parts")
    parser.add_argument("--test_nof_parts", nargs="*", type=int, default=[], help="Test number of parts")
    parser.add_argument("--balanced_train_sampler", action='store_true')
    parser.add_argument("--test_mode", type=str, default="seq", help="Either step, seq, p@k or all")
    parser.add_argument("--skip_acc", type=float, default=0.5, help="Skip folds with accuracy lower than this")

    # NF args
    parser.add_argument("--nf_lr", type=float, default=1e-5, help="NF Learning Rate")
    parser.add_argument("--nf_layers", type=int, default=749, help="NF Number of Layers")
    parser.add_argument("--nf_batch", type=int, default=32, help="NF Batch Size")
    parser.add_argument("--nf_val_set", type=str, help="Validation set path", default="")
    parser.add_argument("--nf_ckpt", type=str, help="NF checkpoint", default="")
    parser.add_argument("--nf_base_dist", type=str, help="Which base distribution to use", default="gaussian")
    parser.add_argument("--nf_supervised_loss", action='store_true', help="Use fully supervised loss formulation")
    parser.add_argument("--nf_c_threshold", type=float, default=100, help="Threshold to remove OoD samples from NLL objective")
    parser.add_argument("--nf_act_norm", action='store_true')

    args = parser.parse_args()
    args.ranges = [tuple(map(int, s.split(',', maxsplit=1))) for s in args.ranges.split()]

    main(args)
