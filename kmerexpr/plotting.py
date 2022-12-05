import matplotlib.pyplot as plt
import os
import numpy as np
from utils import get_errors


def plot_scatter(title,xaxis,yaxis, horizontal = False, save_path="./figures"):
    plt.scatter(xaxis,yaxis , s=5, alpha=0.4 )  #theta_opt
    if horizontal:
        title = title + "-psi-minus-scatter"
        plt.plot([0,np.max(xaxis)], [0,0], '--')
        plt.ylabel(r"$ \psi^{opt} - \psi^{*}$", fontsize=25)
    else :
        max_scal = np.max([np.max(xaxis), np.max(yaxis)])
        title = title + "-psi-scatter"
        plt.plot([0,max_scal], [0,max_scal], '--')
        plt.ylabel(r"$ \psi^{*}$", fontsize=25)

    plt.xlabel(r"$ \psi^{opt}$", fontsize=25)
    plt.savefig(os.path.join(save_path, title + ".pdf"), bbox_inches="tight", pad_inches=0.01)
    print("Saved plot ", os.path.join(save_path, title + ".pdf"))
    plt.close()

def plot_general(result_dict, title, save_path, threshold=False, yaxislabel=r"$ f(x^k)/f(x^0)$", xaxislabel="Effective Passes", 
                xticks=None, logplot=True, fontsize=30, miny = 10000
):
    plt.rc("text", usetex=True)
    plt.rc("font", family="sans-serif")
    palette = ["#377eb8","#ff7f00","#984ea3","#4daf4a","#e41a1c","brown","green","red",]
    markers = ["^-","1-","*-","s-","+-","o-",">-","d-","2-","3-","4-","8-","<-",]
    
    for algo_name, marker, color in zip(result_dict.keys(), markers, palette):
        print("plotting: ", algo_name)
        result = result_dict[algo_name] # result is a 2-d list with different length
        len_cut = len(min(result, key=len))# cut it with min_len and convert it to numpy array for plot
        result = np.array(list(map(lambda arr: arr[:len_cut], result)))
        val_avg = np.mean(result, axis=0)
        if threshold:
            len_cut = (
                np.argmax(val_avg <= threshold) + 1
                if np.sum(val_avg <= threshold) > 0
                else len(val_avg)
            )
            val_avg = val_avg[:len_cut]
        newlength = len(val_avg)
        val_min = np.min(result, axis=0)[:newlength]
        val_max = np.max(result, axis=0)[:newlength]
        # std_result = np.std(result, axis=0)[:newlength]
        # val_min = np.add(val_avg, -std_result)
        # val_max = np.add(val_avg, std_result)
        if xticks is None:
            xticks_p = np.arange(newlength)
        else:
            xticks_p = xticks[:newlength]
        markevery = 1
        if newlength > 20:
            markevery = int(np.floor(newlength / 15))
        if (np.min(val_avg) <= 0 or logplot ==False):  # this to detect negative values and prevent an error to be thrown
            plt.plot(xticks_p, val_avg, marker, markevery=markevery, markersize=12, label=algo_name, lw=3, color=color)
        else:
            plt.semilogy(xticks_p, val_avg, marker, markevery=markevery, markersize=12, label=algo_name, lw=3, color=color)
        # plt.fill_between(xticks_p, val_min, val_max, alpha=0.2, color=color)
        newmincand = np.min(val_min)
        if miny > newmincand:
            miny = newmincand
    plt.ylim(bottom=miny)  # (1- (miny/np.abs(miny))*0.1)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=fontsize)
    plt.xlabel(xaxislabel, fontsize=25)
    plt.ylabel(yaxislabel, fontsize=25)
    # plt.title(title, fontsize=25)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(
        os.path.join(save_path, title + ".pdf"), bbox_inches="tight", pad_inches=0.01
    )
    print("Saved plot ", os.path.join(save_path, title + ".pdf"))
    return plt.gcf()  # or try plt.figure(1)   


def plot_iter(result_dict, problem, title, save_path, threshold=False, tol=False, yaxislabel=r"$ f(x^k)/f(x^0)$", fontsize=30):
    plot_general(
        result_dict=result_dict,
        problem=problem,
        title=title,
        save_path=save_path,
        threshold=threshold,
        tol=tol,
        yaxislabel=yaxislabel,
        fontsize=fontsize,
    )
    
def plot_error_vs_iterations(dict_results, theta_true, title, model_type, save_path="./figures"):
    errors_list = []
    dict_plot = {}
    errors= get_errors(dict_results['xs'], theta_true)
    errors_list.append(errors)
    dict_plot[model_type] = errors_list
    plot_general(dict_plot, title=title , save_path=save_path,
                yaxislabel=r"$\|\theta -\theta^{*} \|$", xticks= dict_results['iteration_counts'], xaxislabel="iterations")
    plt.close()


def plot_general_test(result_dict, title, save_path, threshold=False, yaxislabel=r"$ f(x^k)/f(x^0)$", xaxislabel="Effective Passes", 
                xticks=None, logplot=True, fontsize=30, miny = 10000
):
    ax = plt.gca()
    plt.rc("text", usetex=True)
    plt.rc("font", family="sans-serif")
    # ax.figure(figsize=(9, 8), dpi=1200)
    palette = ["#377eb8","#ff7f00","#984ea3","#4daf4a","#e41a1c","brown","green","red",]
    markers = ["^-","1-","*-","s-","+-","o-",">-","d-","2-","3-","4-","8-","<-",]
    
    for algo_name, marker, color in zip(result_dict.keys(), markers, palette):
        print("plotting: ", algo_name)
        result = result_dict[algo_name] # result is a 2-d list with different length
        len_cut = len(min(result, key=len))# cut it with min_len and convert it to numpy array for plot
        result = np.array(list(map(lambda arr: arr[:len_cut], result)))
        val_avg = np.mean(result, axis=0)
        if threshold:
            len_cut = (
                np.argmax(val_avg <= threshold) + 1
                if np.sum(val_avg <= threshold) > 0
                else len(val_avg)
            )
            val_avg = val_avg[:len_cut]
        newlength = len(val_avg)
        val_min = np.min(result, axis=0)[:newlength]
        val_max = np.max(result, axis=0)[:newlength]
        # std_result = np.std(result, axis=0)[:newlength]
        # val_min = np.add(val_avg, -std_result)
        # val_max = np.add(val_avg, std_result)
        if xticks is None:
            xticks_p = np.arange(newlength)
        else:
            xticks_p = xticks[:newlength]
        markevery = 1
        if newlength > 20:
            markevery = int(np.floor(newlength / 15))
        if (np.min(val_avg) <= 0 or logplot ==False):  # this to detect negative values and prevent an error to be thrown
            ax.plot(xticks_p, val_avg, marker, markevery=markevery, markersize=12, label=algo_name, lw=3, color=color)
        else:
            ax.semilogy(xticks_p, val_avg, marker, markevery=markevery, markersize=12, label=algo_name, lw=3, color=color)
        # plt.fill_between(xticks_p, val_min, val_max, alpha=0.2, color=color)

        newmincand = np.min(val_min)
        if miny > newmincand:
            miny = newmincand
    ax.ylim(bottom=miny)  # (1- (miny/np.abs(miny))*0.1)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=fontsize)
    ax.xlabel(xaxislabel, fontsize=25)
    ax.ylabel(yaxislabel, fontsize=25)
    # plt.title(title, fontsize=25)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ax.savefig(
        os.path.join(save_path, title + ".pdf"), bbox_inches="tight", pad_inches=0.01
    )
    print("Saved plot ", os.path.join(save_path, title + ".pdf"))
    return plt.gcf() 