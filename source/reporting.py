import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/.configs/" # workaround for pyplot config dir
import matplotlib.pyplot as plt

from collections import defaultdict
import sklearn.metrics as sk_metrics
import numpy as np


def save_stat_boxplot(stats, parts_list, args):
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)


    for n in parts_list:
        plt.clf()
        for s in stats[n].keys():
            if stats[n][s]["action"] != "box":
                continue
            
            res_dict, _ = stats[n][s]["func"].compute()

            labels, data = res_dict.keys(), res_dict.values()

            bp = plt.boxplot(data,  sym='')
            set_box_color(bp, stats[n][s]["color"])

            plt.plot([], c=stats[n][s]["color"], label=s)

        plt.legend()
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylim(0, 1)
    
        #plt.title('Initial Failed State')
        plt.title('Candidate Function Values, %d parts' % n)
        plt.savefig("functions_%d.png" % n)


def save_histogram(stats, parts_list):
    save = False

    plt.clf()
    for n in parts_list:
        for s in stats[n].keys():
            if stats[n][s]["action"] != "hist":
                continue
            
            save = True
            res_dict, _ = stats[n][s]["func"].compute()
            plt.hist(res_dict["prod"], bins=40, label="%d: %s" % (n, s), alpha=0.5, density=True, range=(0, 1))

    if not save:
        return 

    plt.legend()
    plt.savefig("Histogram.png")


def print_stats(stats, parts_list):
    prt = False
    avg_stats = defaultdict(int)
    for n in parts_list:
        for s in stats[n].keys():
            if stats[n][s]["action"] != "print":
                continue

            prt = True

            val = stats[n][s]["func"].compute().item()
            print("%d parts %s: " % (n, s), val)
            avg_stats[s] += val

    if not prt:
        return 

    if len(avg_stats.keys()):
        print("Averages: ")
        for s in avg_stats.keys():
            print(s , avg_stats[s] / len(parts_list))


def save_pr_curve(stats, parts_list):
    save = False

    plt.clf()
    for n in parts_list:
        for s in stats[n].keys():
            if stats[n][s]["action"] != "curve":
                continue

            save = True

            ps, rs, _ = stats[n][s]["func"].compute()
            disp = sk_metrics.PrecisionRecallDisplay(ps, rs)

            label = "%d parts" % n
            if s == "curve":
                label += ", AUC=%.4f" % sk_metrics.auc(rs, ps)
            else:
                label += ", %s" % s
            disp.plot(ax=plt.gca(), name=label)
        
    if not save:
        return
    
    plt.title('Precision-Recall Curve')
    plt.xlim((0, 1))
    plt.savefig("pr_curve.png")


def save_roc_auc_curve(feasibility_preds, feasibility_gt):
    plt.clf()
    for n in feasibility_preds.keys():
        sk_metrics.RocCurveDisplay.from_predictions(np.array(feasibility_gt[n]), np.array(feasibility_preds[n]), ax=plt.gca(), name="%d parts" % n)

    plt.title('RoC Curve')
    plt.xlim((0, 1))
    plt.savefig("roc_curve_feasibility.png")

