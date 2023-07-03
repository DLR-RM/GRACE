import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pytorch_lightning as pl
import random
import pickle


assembly_filenames = {
    "spec": "assembly.spec",
    "dist": "distances.csv", 
    "sol": "solutions.csv"
}

part_to_surfaces = {
    "item_profile_200": {
        "surface_1": "short",
        "surface_2": "long",
        "surface_3": "short",
        "surface_4": "long"
    },
    "item_profile_300": {
        "surface_1": "short",
        "surface_2": "long",
        "surface_3": "short",
        "surface_4": "long"
    },
    "item_angle_bracket": {
        "surface_1": "lateral",
        "surface_2": "lateral"
    }
}

part_to_num = {
    "item_profile_200": 0,
    "item_profile_300": 1, 
    "item_angle_bracket": 2
}
part_and_surface_to_num = {
    ("item_profile_200", "short"): 0,
    ("item_profile_200", "long"): 1,
    ("item_profile_300", "short"): 2,
    ("item_profile_300", "long"): 3,
    ("item_angle_bracket", "lateral"): 4
}


def get_dirlist(root_dir):
    dir_list = list()
    with os.scandir(root_dir) as rit:
        for entry in rit:
            if not entry.name.startswith('.') and entry.is_dir():
                dir_list.append(entry.path)

    return dir_list


def is_usable_assembly(dir_path):
    if not os.access(dir_path, os.R_OK):
        return False

    file_list = os.listdir(dir_path)

    for name in assembly_filenames.values():
        if name not in file_list:
            return False

    return True


def mat_string_to_array(mat):
    if mat.count('&') != 3:
        print(mat)
        raise ValueError("Something wrong with RT matrix!")
    
    mat = mat.replace('&', ';')
    if mat.count(';') != 15:
        print(mat)
        raise ValueError("Something wrong with RT matrix!")

    arr = np.fromstring(mat, sep=';', dtype=float)
    return arr


def get_parts_dist(mat1, mat2):
    x1, x2 = mat1[3], mat2[3]
    y1, y2 = mat1[7], mat2[7]

    dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    return dist


def read_spec_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        if not lines[0].startswith('begin Pieces') or not lines[len(lines) - 1].startswith('end Pieces'):
            raise ValueError('Malformed spec file.')

        parts = dict()
        part_id, surface_id = 0, 0
        for i in range(1, len(lines) - 1):
            words = lines[i].split()

            part_name = words[0]
            part_type = words[1]
            surfaces = dict()
            for surface_name, surface_type in part_to_surfaces[part_type].items():
                surfaces[surface_name] = {
                    "id": surface_id,
                    "type": surface_type
                }
                surface_id += 1

            part = {
                "id": part_id,
                "type": part_type,
                "matrix": mat_string_to_array(words[2]),
                "surfaces": surfaces
            }
            parts[part_name] = part
            part_id += 1

    return parts


def split_part_surface(item):
    part, surface_num = item.split("_surface_")
    surface = "surface_" + surface_num

    return part, surface


def read_dist_file(file_path):
    distances = list()

    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)

        for line in csvreader:
            part_1, surface_1 = split_part_surface(line[0])
            part_2, surface_2 = split_part_surface(line[1])
            dist = math.floor(float(line[2]))
            distances.append({
                "p1": part_1, "s1": surface_1, "p2": part_2, "s2": surface_2, "dist": dist
            })

    return distances


def read_sol_file(file_path):
    solutions = list()

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.rstrip().split(',')
            solutions.append(parts)

    return solutions


def read_assembly_files(dir):
    spec_filepath = os.path.join(dir, assembly_filenames["spec"])
    parts = read_spec_file(spec_filepath)

    dist_filepath = os.path.join(dir, assembly_filenames["dist"])
    distances = read_dist_file(dist_filepath)

    sol_filepath = os.path.join(dir, assembly_filenames["sol"])
    solutions = read_sol_file(sol_filepath)

    return parts, distances, solutions


def get_node_features(parts):
    p_features, s_features, p_positions = list(), list(), list()

    for p in parts.values():
        pf = {
            "assembled": 0,
            "part_type": part_to_num[p["type"]],
            'part_id': p["id"],
            }
        p_features.append(pf)
        
        x_pos = p['matrix'][3]
        y_pos = p['matrix'][7]
        p_positions.append([x_pos, y_pos])


        for s in p["surfaces"].values():
            sf = {
                "surface_type": part_and_surface_to_num[(p["type"], s["type"])],
                'surface_id': s['id'],
                #'matrix': p['matrix']
            }
            s_features.append(sf)

    return p_features, s_features, p_positions


def get_edge_features(parts, distances, part_distances=False):
    s_to_s = {
        "dist": {"edge_index": list(), "edge_weight": list()},
    }

    for item in distances:
        p1 = parts[item["p1"]]
        s1 = p1["surfaces"][item["s1"]]
        p2 = parts[item["p2"]]
        s2 = p2["surfaces"][item["s2"]]

        s_to_s["dist"]["edge_index"].append([s1["id"], s2["id"]])
        s_to_s["dist"]["edge_weight"].append(item["dist"])

    # connect all surfaces of the same part
    for p in parts.values():
        already_connected = set()
        for s1 in p["surfaces"].values():
            for s2 in p["surfaces"].values():
                if s1["id"] != s2["id"] and (s2["id"], s1["id"]) not in already_connected:
                    s_to_s["dist"]["edge_index"].append([s1["id"], s2["id"]])
                    s_to_s["dist"]["edge_weight"].append(1)

    # connect each part to all its surfaces
    s_to_p = {"to": {"edge_index": list()}}
    for p in parts.values():
        for s in p["surfaces"].values():
            s_to_p["to"]["edge_index"].append([s["id"], p["id"]])

    if not part_distances:
        p_to_p = None
    else:
        p_to_p = {
            "dist": {"edge_index": list(), "edge_weight": list()}
        }

        for p1 in parts.values():
            for p2 in parts.values():
                p_to_p["dist"]["edge_index"].append([p1["id"], p2["id"]])
                p_to_p["dist"]["edge_weight"].append(get_parts_dist(p1["matrix"], p2["matrix"]))

    return s_to_s, s_to_p, p_to_p


def set_assembled(parts, assembled_parts, p_features, s_features):
    for part_name in parts.keys():
        part_id = parts[part_name]["id"]
        p_features[part_id]["assembled"] = 1 if part_name in assembled_parts else 0

    return p_features, s_features


def create_labels(parts, next_parts, mode="NML"):
    """
    format takes one of the following values:
    "duplicates": Allow multiple assembly sequences with the same prefix
    which continue with a different single "next_part" with probability 100%.
    [[0, 0, 1, 0 , 0], [0, 0, 0, 1 ,0]] -> CE
    "distribution": Merge all sequences with the same prefix and output multiple possible "next_parts",
    while normalizing the output to a distribution. [0, 0, 0.5, 0.5 , 0] -> CE
    "multilabel": Similar to distribution, but without normalization [0, 0, 1, 1 , 0] -> BCE
    """
    if mode == "NSL":
        label_format = "duplicates"
    elif mode == "NML":
        label_format = "multilabel"
    else:
        raise ValueError("Unexpected mode argument")

    labels = list()
    if label_format in ["distribution", "multilabel"]:
        # return a single label list
        label = np.zeros(len(parts))

        for name in next_parts:
            part_id = parts[name]["id"]
            label[part_id] = 1

        if label_format == "distribution":
            nof_possible_parts = len(next_parts)
            label /= nof_possible_parts

        labels.append(label)

    else:
        # return a label list per part
        for name in next_parts:
            label = np.zeros(len(parts))

            part_id = parts[name]["id"]
            label[part_id] = 1

            labels.append(label)

    return labels


def dict_values_to_tensor(d):
    # treat also features which are arrays themselves
    vec = np.stack(list(np.hstack(np.array(list(p.values()), dtype=object)) for p in d), axis=0)
    res = torch.tensor(vec, dtype=torch.float)
    return res


def dict_values_to_index(d):
    # get a single number, to be used as index for embedding, from a list of features

    f = list()
    for p in d:
        str_list = [str(n) for n in p.values()]
        idx = int(''.join(str_list))
        f.append(idx)

    res = torch.tensor(f, dtype=torch.int).view(len(d), 1)

    return res


def build_label_graph(p_features, s_features, s_to_s, s_to_p, p_labels, p_to_p=None, p_positions=None):
    data = HeteroData()

    data["part"].x = dict_values_to_tensor(p_features)
    data["part"].y = torch.tensor(p_labels, dtype=torch.float).reshape(-1, 1)

    if p_positions:
        data["part"].position = torch.tensor(np.stack(p_positions), dtype=torch.float)

    data["surface"].x = dict_values_to_tensor(s_features)


    """
    to_undirected() only works with identical src and dst nodes.
    T.ToUndirected() doesn't support edge_attributes.
    We use both...
    """

    # surface <-> surface
    for rel in s_to_s.keys():
        indices, weights = to_undirected(
            edge_index=torch.tensor(s_to_s[rel]["edge_index"], dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(s_to_s[rel]["edge_weight"], dtype=torch.float),
            reduce="mean"
        )
        data['surface', rel, 'surface'].edge_index = indices
        data['surface', rel, 'surface'].edge_weight = weights

    # surface -> part
    indices = torch.tensor(s_to_p["to"]["edge_index"], dtype=torch.long).t().contiguous()
    data['surface', 'to', 'part'].edge_index = indices

    # part <-> part
    if p_to_p:
        indices, weights = to_undirected(
            edge_index=torch.tensor(p_to_p["dist"]["edge_index"], dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(p_to_p["dist"]["edge_weight"], dtype=torch.float),
            reduce="mean"
        )
        data['part', 'dist', 'part'].edge_index = indices
        data['part', 'dist', 'part'].edge_weight = weights

    return data


def build_single_class_graph(s_features, s_to_s, label):
    data = HeteroData()

    data.y = torch.tensor(label, dtype=torch.float).view(1, 1)

    data["surface"].x = dict_values_to_index(s_features)

    # surface <-> surface
    for rel in s_to_s.keys():
        indices, weights = to_undirected(
            edge_index=torch.tensor(s_to_s[rel]["edge_index"], dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(s_to_s[rel]["edge_weight"], dtype=torch.float),
            reduce="mean"
        )
        data['surface', rel, 'surface'].edge_index = indices
        data['surface', rel, 'surface'].edge_weight = weights

    return data


def get_assembly_sequences(solutions, nof_parts):
    # key in sequences is set of already assembled parts, value is set of possible next parts
    sequences = defaultdict(set)

    for sol in solutions:
        for i in range(0, len(sol)):
            assembled_parts = frozenset(sol[:i])
            next_part_name = sol[i]
            sequences[assembled_parts].add(next_part_name)

        '''
        Check if this is an intermediate solution, meaning, it has fewer parts then the assembly.
        In this case, we should add another step, with the maximal number of parts, leading to the failure,
        meaning, no part is possible next.
        '''
        if len(sol) < nof_parts:
            sequences[frozenset(sol)] = set()

    return sequences


def create_single_class_graph_objects(parts, distances, solutions, assembly_name, opts):

    _, s_features, _ = get_node_features(parts)
    s_to_s, _, _= get_edge_features(parts, distances, opts.part_distances)

    label = 1 if len(solutions) else 0

    data = build_single_class_graph(s_features, s_to_s, label)

    return [(1, 1, data)]


def create_label_graph_objects(parts, distances, solutions, assembly_name, part_distances):
    output = list()

    p_features, s_features, _ = get_node_features(parts)
    s_to_s, s_to_p, p_to_p = get_edge_features(parts, distances, part_distances)

    sequences = get_assembly_sequences(solutions, len(parts))
    for i, (assembled_parts, possible_next_parts) in enumerate(sequences.items()):
        p_features, s_features = set_assembled(parts, assembled_parts, p_features, s_features)
        p_labels = create_labels(parts, possible_next_parts, mode= "NML")

        # separate graph per labels sequence
        for j, labels in enumerate(p_labels):
            data = build_label_graph(p_features, s_features, s_to_s, s_to_p, labels, p_to_p)

            output.append((i, j, data))

    # for infeasible graphs we don't have a solution sequence
    if len(solutions) == 0:
        p_labels = create_labels(parts, [], mode= "NML")
        data = build_label_graph(p_features, s_features, s_to_s, s_to_p, p_labels[0], p_to_p)
        output.append((1, 1, data))

    return output


def create_sequence_graph(parts, distances, solutions, assembly_name, part_distances=False, part_positions=False):

    p_features, s_features, p_positions = get_node_features(parts)
    s_to_s, s_to_p, p_to_p = get_edge_features(parts, distances, part_distances=part_distances)

    data = build_label_graph(p_features, s_features, s_to_s, s_to_p, p_labels=[0], p_to_p=p_to_p, p_positions=p_positions if part_positions else None)

    sequences = list()
    if len(solutions):
        for sol in solutions:
            # add sequence only if it's complete, meaning not intermediate
            if len(sol) == len(parts):
                sequences.append([parts[part_name]["id"] for part_name in sol])

    # only overwrite y if we added something
    if len(sequences):
        data["part"].y = sequences
    # another wounderfull hack, since dataloader doesn't support empty lists
    else:
        data["part"].y = [['dummy']]
    
    data['file'].name = assembly_name
    
    return [(1, 1, data)]


def create_sequence_graph_failed_states(parts, distances, solutions, assembly_name, opts):
    output = list()

    p_features, s_features, _ = get_node_features(parts)
    s_to_s, s_to_p, _ = get_edge_features(parts, distances)

    # in failed state, the final solution appearing in file is the list of assembled parts leading to failure
    for i, assembled_parts in enumerate(solutions):
        if len(assembled_parts) >= len(parts):
           raise ValueError("Infeasible assemblies can't have so many parts..")

        p_features, s_features = set_assembled(parts, assembled_parts, p_features, s_features)
        data = build_label_graph(p_features, s_features, s_to_s, s_to_p, p_labels=[0])

        output.append((i, 1, data))

    return output

def read_raw_dir_list(dir_list, ranges=None):
    out = list()

    for dir in tqdm(dir_list):
        if not is_usable_assembly(dir):
            continue
        
        dir_index = int(os.path.basename(dir))

        if ranges is not None:
            if all([(dir_index < range_min or dir_index > range_max) for (range_min, range_max) in ranges]):
                continue

        assembly_name = os.path.basename(dir)
        parts, distances, solutions = read_assembly_files(dir)
        
        if len(parts) == 0 or len(distances) == 0:
            raise ValueError("Something wrong with assembly %s" % assembly_name)

        # if no solutions are given, use intermediate steps intead if it's given
        if len(solutions) == 0:
            intermediate_sol_filepath = os.path.join(dir, "inter_solutions.csv")            
            
            if os.path.exists(intermediate_sol_filepath):
                solutions = read_sol_file(intermediate_sol_filepath)       

        out.append((assembly_name, parts, distances, solutions))

    print("%d assemblies chosen after range" % len(out))

    return out


def read_raw_files(raw_dir, opts):
    dir_list = get_dirlist(raw_dir)

    random.shuffle(dir_list)

    train_len = int(opts.train_ratio * len(dir_list))
    val_len = int(opts.val_ratio * len(dir_list))

    print("Creating dataset split for %s..." % opts.split)

    if opts.split == "all":
        dir_list = dir_list[0:len(dir_list)]
    elif opts.split == "train":
        dir_list = dir_list[0:train_len]
    elif opts.split == "val":
        dir_list = dir_list[train_len:train_len + val_len]
    elif opts.split == "trainval":
        dir_list = dir_list[0:train_len + val_len]
    elif opts.split == "test":
        dir_list = dir_list[train_len + val_len: len(dir_list)]
    else:
        raise ValueError("Unrecognized dataset split %s" % opts.split)

    print("%d files chosen" % len(dir_list))

    return read_raw_dir_list(dir_list, opts.ranges)
    

def remove_files(dir):
    if not os.path.exists(dir):
        return

    files_in_directory = os.listdir(dir)
    files = [file for file in files_in_directory if file.endswith(".pt")]
    for file in files:
        os.remove(os.path.join(dir, file))


def in_which_set(args):
    filter_train = args.train_nof_parts
    filter_val = args.val_nof_parts

    set_def = {p: "train" for p in range(3, 8)}
    
    # no filter, everything in
    if filter_train is None or len(filter_train) == 0:
        return set_def
    
    for p in set_def.keys():
        if p not in filter_train:
            set_def[p]= None

    if filter_val is None or len(filter_val) == 0:
        return set_def

    for p in set_def.keys():
        if p in filter_val:
            set_def[p]= "val"

    return set_def


def create_sets(assemblies, train_index, test_index, args):
    
    # trainval sets
    set_def = in_which_set(args)
    
    print("Provided split between train/val: ", set_def)
    if args.val_nof_parts is None or len(args.val_nof_parts) == 0:
        print("Will split train/val based on percantage later on..")

    trainval_set = list()
    for i in tqdm(train_index):
        assembly_name, parts, distances, solutions = assemblies[i]

        if set_def[len(parts)] is None:
            continue

        graphs = create_label_graph_objects(parts, distances, solutions, assembly_name, part_distances=args.aux_loss)

        graphs = [(len(parts), data) for (i, j, data) in graphs] # walkaround, keep record of nof parts
        trainval_set += graphs

    train_index = list()
    val_index = list()
    for i, (nof_parts, _) in enumerate(trainval_set):
        if set_def[nof_parts] == "train":
            train_index.append(i)
        elif set_def[nof_parts] == "val":
            val_index.append(i)
    trainval_set = [data for (_, data) in trainval_set] # get rid of nof parts

    # test set
    test_set_seq, test_set_step = list(), list()
    filter_nof_parts = args.test_nof_parts
    for i in tqdm(test_index):
        assembly_name, parts, distances, solutions = assemblies[i]

        if filter_nof_parts is not None and len(filter_nof_parts) and len(parts) not in filter_nof_parts:
            continue

        graphs = create_sequence_graph(parts, distances, solutions, assembly_name, part_distances=args.aux_loss)
        test_set_seq += [data for (i, j, data) in graphs]

        graphs = create_label_graph_objects(parts, distances, solutions, assembly_name, part_distances=args.aux_loss)
        test_set_step += [data for (i, j, data) in graphs]

    return trainval_set, test_set_seq, test_set_step, train_index, val_index


def read_filter_assemblies(args):
    # first, parse raw files
    if not args.assemblies_cache or not os.path.exists(args.assemblies_cache):
        print("Parsing raw assemblies...")
        
        dir_list = get_dirlist(args.raw_dir)
        assemblies = read_raw_dir_list(dir_list, args.ranges)
        with open(args.assemblies_cache, 'wb') as handle:
            pickle.dump(assemblies, handle)
    else:
        print("Getting assemblies from cache...")
        with open(args.assemblies_cache, 'rb') as handle:
            assemblies = pickle.load(handle)

    print("Total input assemblies: ", len(assemblies))
    
    # filter infeasbile
    output = list()
    for (assembly_name, parts, distances, solutions) in assemblies:
        # skip infeasible without intermediate steps
        if len(solutions) == 0 and args.infeasible in ["exclude_all", "intermediate_only"]:
            continue

        # skip if any solution is shorter than number of parts, its intermediate steps
        if len(solutions) and len(solutions[0]) < len(parts) and args.infeasible in ["exclude_all", "no_step_only", "convert_to_no_step"]:
            # convert to not step..
            if args.infeasible == "convert_to_no_step":
                solutions = list()
            else:
                continue

        output.append((assembly_name, parts, distances, solutions))

    print("Total assemblies after infeasible filtering: ", len(output))

    return output


def create_dataset_folder(input_dir, output_dir, opts):
    pl.seed_everything(32)

    output_dir_raw = os.path.join(output_dir, "raw/")

    if not os.path.isdir(output_dir_raw):
        print("Creating dataset directory")
        os.makedirs(output_dir_raw, mode=0o770, exist_ok=True)

    files_in_directory = os.listdir(output_dir_raw)
    if opts.recreate_data:
        remove_files(output_dir_raw)
        # remove also artifacts created by torch dataloader
        remove_files(os.path.join(output_dir, "processed/"))
        
    elif len(files_in_directory):
        print("Dataset already created.")
        return

    print("Parsing raw files...")
    processed = read_raw_files(input_dir, opts)
    
    total_count = 0
    nof_part_count = defaultdict(lambda: 0)

    print("Building graphs...")
    part_types = {n: defaultdict(int) for n in range(3,8)}
    for (assembly_name, parts, distances, solutions) in tqdm(processed):

        nof_parts = len(parts)
        if len(opts.nof_parts) and nof_parts not in opts.nof_parts:
            continue

        # infeasible without intermediate steps
        if len(solutions) == 0 and opts.infeasible in ["exclude_all", "intermediate_only"]:
            continue

        # if any solution is shorter than number of parts, its intermediate steps
        elif len(solutions) and len(solutions[0]) < nof_parts and opts.infeasible in ["exclude_all", "no_step_only", "convert_to_no_step"]:
            # convert to not step..
            if opts.infeasible == "convert_to_no_step":
                solutions = list()
            else:
                continue

        for p in parts.values():
            part_types[nof_parts][p["type"]] += 1

        # a single assembly may be repressed with several graphs
        graphs = list()
        if opts.mode == "NML" or opts.mode == "NSL":
            graphs = create_label_graph_objects(parts, distances, solutions, assembly_name, opts.part_distances)
        elif opts.mode == "GSC":
            graphs = create_single_class_graph_objects(parts, distances, solutions, assembly_name, opts)
        elif opts.mode == "GSEQ":
            graphs = create_sequence_graph(parts, distances, solutions, assembly_name, opts.part_distances, opts.part_positions)

        for (i, j, data) in graphs:
            filename = f'data_{assembly_name}_{i}_{j}.pt'
            torch.save(data, os.path.join(output_dir_raw, filename))

        nof_graphs = len(graphs)
        nof_part_count[nof_parts] += nof_graphs

        total_count += nof_graphs
        if opts.nof_samples and total_count >= opts.nof_samples:
            break

    print("Total %d graphs created" % total_count)
    print("Part breakdown: ", nof_part_count.items())
    #print(part_types)


def main(args):
    args.ranges = [tuple(map(int, s.split(',', maxsplit=1))) for s in args.ranges.split()]

    create_dataset_folder(args.raw_data, args.dataset_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Dataset")

    parser.add_argument("--raw_data", type=str, default="/home/rodr_is/data_matan/", help="Path to assembly folders")
    parser.add_argument("--dataset_path", type=str, default="/home/atad_ma/Projects/datasets/gseq_7_test/",
                        help="Dataset path")
    
    parser.add_argument("--mode", type=str, default="GSEQ",
                        help="One of GSC (Graph Single Class), NML (Node Multi Label), NSL (Node Single Label), GSEQ (Graph sequences)")

    parser.add_argument("--ranges", type=str, nargs='*', default="0,19999 70000,71000", help="Index range for assembly directory")
    parser.add_argument("--nof_parts", nargs="*", type=int, default=[7], help="Create only graphs with this number of parts")
    parser.add_argument("--nof_samples", type=int, help="Number of samples to process")
    parser.add_argument("--debug_print", action='store_true', help="Print created graphs")
    parser.add_argument("--recreate_data", action='store_true', help="Delete previous files and recreate them")

    parser.add_argument("--infeasible", type=str, default="exclude_all", help="Either exclude_all, intermediate_only, no_step_only, convert_to_no_step, include_all")
    parser.add_argument("--part_distances", action='store_true', default=False, help="Create part distances edges")
    parser.add_argument("--part_positions", action='store_true', default=False, help="Create part distances edges")

    parser.add_argument("--train_ratio", type=float, default=0.75, help="Train set size")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val set size")
    parser.add_argument("--split", type=str, default="test", help="Train set size")

    args = parser.parse_args()
    main(args)
