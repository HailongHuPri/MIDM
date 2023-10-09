import numpy as np

        
def get_metrics(label, score, fixed_fpr=0.001):
    """
    Compute TPR at FPR
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(label, score)

    
    tpr_at_low_fpr = tpr[np.where(fpr <= fixed_fpr)[0][-1]]
    
    
    return tpr_at_low_fpr


def plot_all_steps(prediction,save_path=None):
    '''
    For loss based attacks.
    TPR at low FPR for all diffusion steps.
    '''
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("darkgrid")
    palette_tab = plt.get_cmap('tab10')
    color_alfa = 1
    
    x_idx = np.arange(0,1000,1)

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('font', size=15)        

    plt.plot(x_idx, prediction[:,0], linewidth=2, label='TPR@10%FPR',color=palette_tab(0))
    plt.plot(x_idx, prediction[:,1], linewidth=2, label='TPR@1%FPR',color=palette_tab(1))
    plt.plot(x_idx, prediction[:,2], linewidth=2, label='TPR@0.1%FPR',color=palette_tab(2))
    plt.plot(x_idx, prediction[:,3], linewidth=2, label='TPR@0.01%FPR',color=palette_tab(4))

    

    plt.ylim([0, 1.05])
    # axis labels
    plt.xlabel('Diffusion Steps',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # show the legend
    plt.legend(fontsize=14)
    # show the plot
    if save_path != None:
        plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0.01,dpi=300) 
    plt.show()
    

def plot_one_step(labels, predictions,save_path=None):
    '''
    For loss based attacks.
    ROC curve for one step.
    '''
    from sklearn.metrics import roc_curve
    import seaborn as sns
    sns.set_style("darkgrid")
    
    fpr_0, tpr_0, _ = roc_curve(labels, predictions[0])
    fpr_1, tpr_1, _ = roc_curve(labels, predictions[1])
    fpr_2, tpr_2, _ = roc_curve(labels, predictions[2])
    fpr_3, tpr_3, _ = roc_curve(labels, predictions[3])
    fpr_4, tpr_4, _ = roc_curve(labels, predictions[4])
    fpr_5, tpr_5, _ = roc_curve(labels, predictions[5])
    

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(labels))]
    ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs)

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('font', size=15)    
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='',c = 'grey')

    plt.plot(fpr_0, tpr_0, linewidth=2, label='t=0')
    plt.plot(fpr_1, tpr_1, linewidth=3, label='t=200')
    plt.plot(fpr_2, tpr_2, linewidth=2, label='t=500')
    plt.plot(fpr_3, tpr_3, linewidth=2, label='t=600')
    plt.plot(fpr_4, tpr_4, linewidth=2, label='t=800')
    plt.plot(fpr_5, tpr_5, linewidth=2, label='t=999')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([1e-5, 1])
    plt.ylim([1e-5, 2.0])
    # axis labels
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # show the legend
    plt.legend(fontsize=14)
    if save_path!=None:
        plt.savefig(save_path,bbox_inches = 'tight',pad_inches = 0.03,dpi=300)          
    # show the plot
    plt.show()


def plot_TPR_FPR(labels, predictions,save_path=None):
    '''
    For likelihood based attacks.
    ROC curve.
    '''    
    from sklearn.metrics import roc_curve
    import seaborn as sns
    sns.set_style("darkgrid")
    
    fpr_0, tpr_0, _ = roc_curve(labels, predictions)

    

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(labels))]
    ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs)

    import matplotlib.pyplot as plt

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='',c = 'grey')

    plt.plot(fpr_0, tpr_0, linewidth=2, label='Likelihood based attacks')

    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim([1e-5, 1])
    plt.ylim([1e-5, 1])
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    if save_path!=None:
        plt.savefig(save_path,bbox_inches = 'tight',pad_inches = 0)       
    # show the plot
    plt.show()
