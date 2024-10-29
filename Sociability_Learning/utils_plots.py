import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
import scikit_posthocs as sp


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def get_bootstrap_ci(data, ci=95):
    n_bootstrap_samples = 1000
    # Bootstrap resampling
    bootstrap_medians = []
    for _ in range(n_bootstrap_samples):
        # Resample with replacement from the original data
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the median of the bootstrap sample
        bootstrap_median = np.median(bootstrap_sample)
        bootstrap_medians.append(bootstrap_median)
    # Calculate the 95% confidence interval
    confidence_interval = np.percentile(bootstrap_medians, [(100-ci)/2, 100-((100-ci)/2)])

    return confidence_interval

def extract_name(genotype):
    return genotype.split('_')[-1]

def kruskal_test(data):
    formatted_data = [data.loc[ids, 'Prop'].values for ids in data.groupby('Genotype').groups.values()]
    result_kw = kruskal(*formatted_data)

    return result_kw[1]

def multiple_comparisons_test(data, control_line, pval):
    from scipy.stats import kruskal, ranksums
    from statsmodels.stats.multitest import multipletests
   
    #control_group_all = data[data["Genotype"] == control_line]["Prop"]
    #subset = random.sample(control_group_all.index.to_list(), 12)
    #control_group = control_group_all.copy()[subset]

    control_group = data[data["Genotype"] == control_line]["Prop"]
    
    # Extract experimental groups
    experimental_groups = {name: group["Prop"] for name, group in data.groupby("Genotype") if name != control_line}

    # Perform pairwise comparisons between control and experimental groups
    pairwise_results = {}
    for exp_group, values in experimental_groups.items():
        # Perform pairwise comparison using rank-sum test
        #_, p_value = ranksums(control_group, values)
        _, p_value = mannwhitneyu(control_group, values)
        # Store p-value
        pairwise_results[exp_group] = p_value

    # Adjust p-values for multiple comparisons using Bonferroni correction
    adjusted_p_values = multipletests(list(pairwise_results.values()), method='holm')[1]

    conover_adjusted_df = pd.DataFrame(zip(pairwise_results.keys(), adjusted_p_values, pairwise_results.values()),columns=["Genotype","p-value","p-value (wo correction)"])

    conover_adjusted_df = conover_adjusted_df.append({"Genotype": control_line, "p-value":1.1, "p-value (wo correction)":1.1}, ignore_index=True)
    #print(conover_adjusted_df)

    return conover_adjusted_df
    # Print significant pairwise comparisons
    #for exp_group, adj_p_value in zip(pairwise_results.keys(), adjusted_p_values):
    #    if adj_p_value < pval:
    #        print(f"Significant difference between control and {exp_group}: p-value = {adj_p_value}")
        #else:
        #    print("No difference")


def get_statistics_screen(auc_df, screen, pval=0.05, min_samples=8):
    screen_df = auc_df[auc_df['Genotype'].str.contains(screen)]

    control_group_substring = "Empty-split"
    control_data = screen_df[screen_df['Genotype'].str.contains(control_group_substring)]
    control_full_name = control_data['Genotype'].unique()[0]

    ci_vals = get_bootstrap_ci(control_data['Prop'].values)

    screen_median = screen_df.groupby('Genotype').agg({'Prop': ['median', 'count']}).reset_index()
    screen_median.columns = ['Genotype', 'Prop', 'Count']

    selected_lines_medians = screen_median.loc[screen_median["Count"]>=min_samples]

    merged_df = pd.merge(screen_df, screen_median[['Genotype', 'Count']], on='Genotype')
    selected_lines_all = merged_df[merged_df['Count'] >= min_samples]
    
    result_kw = kruskal_test(selected_lines_all)
    #print(result_kw)
    #conover_results = sp.posthoc_conover(selected_lines_all, val_col='Prop', group_col='Genotype', p_adjust="bonferroni")

    conover_adjusted = multiple_comparisons_test(selected_lines_all, control_full_name, pval)
    
    selected_lines_medians = pd.merge(screen_median.loc[screen_median["Count"]>=min_samples], conover_adjusted, on='Genotype')
    selected_lines_medians["Simple-gen"] = selected_lines_medians["Genotype"].apply(extract_name)

    #hits = selected_lines_medians.loc[(selected_lines_medians['p-value']==1.0)|(selected_lines_medians['p-value']<pval)]
    hits = selected_lines_medians.loc[selected_lines_medians['p-value']<pval]
    
    return selected_lines_medians, hits, ci_vals


def get_pval_index(screen_df, limits):
    idx_pval = {}
    if not isinstance(limits, list):
        pvals = [limits]
    else:
        pvals = limits

    all_sorted_pval = screen_df.sort_values(by='p-value', ascending=False).reset_index()

    for pval in pvals:
        idx_pval[f"{pval}"] = all_sorted_pval.index[all_sorted_pval['p-value'] < pval][0] - 0.5

    return all_sorted_pval, idx_pval


def get_order_by_median(data_df, limits={}, ascending=False):
    order=[data_df.loc[0]['Genotype']]
    lim_idx = [1]
    for pval, idx in limits.items():
        lim_idx.append(int(idx+0.5))
    lim_idx.append(len(data_df))

    for low_lim, high_lim in zip(lim_idx[:-1], lim_idx[1:]):
        subset_df = data_df[low_lim:high_lim].copy()
        order_by_median = subset_df.sort_values(by='Prop', ascending=ascending)
        order.extend(order_by_median['Genotype'].to_list())

    df_lines = pd.read_csv('all_lines_screening.csv', header=0)
    brain_regions=[]
    for line in order:
        simple_name = line.split("_")[1]
        region = df_lines.loc[df_lines["driver_line"].str.contains(simple_name)]["brain_region"].values[0]
        brain_regions.append(region)

    return order, brain_regions
        

def plot_single_metric(all_events, x, y, hue=None, use_bins=False, order=[], hue_order=None, horizontal=False, size=6, notch=False, split_flies=False, brain_regions=[], p_values={}, ci=[], save_fig=False,name_fig="", size_fig=(16,9)):
    fig, ax = plt.subplots(figsize=size_fig)
    new_order = order
    color_regions = {"Control":(0,0,0),
                     "ALIN":(1,0,1),
                     "ALPN":(1,0,1),
                     "DN":(225/255,180/255,120/255),
                     "LH":(150/255,0,1),
                     "MBIN":(0,1,1),
                     "MBON":(0,0,1),
                     "MBEN":(0,1,0),
                     "NM":(1,123/255,0),
                     "NP":(125/255,1,0),
                     "other":(200/255,200/255,200/255),
                     "VPN":(1,0,0)
                     }

    if len(brain_regions)>0:
        palette = [color_regions[region] for region in brain_regions]
    else:
        palette=None
    
    if len(new_order)>0:
        events = all_events[all_events[x].isin(new_order)]
    else:
        events = all_events.copy()   

    genotypes = np.unique(np.array(events[x].values))
    bins = np.unique(np.array(events["Bin"].values))
    #print(bins)
    #print(genotypes)
    for gen in genotypes:
        num_flies = int(len(events.loc[events[x]==gen])/len(bins))
        try:
            screen = gen.split("_")[0].split("-")[-1]
            gal4 = gen.split("_")[1]
        except:
            screen = ""
            gal4 = gen.split("_")[1]
        if horizontal or len(new_order)>20:
            new_name = f"{gal4}"
        else:
            if screen not in "":
                new_name = f"{gal4}\n{screen}\n(n={num_flies})"
                #new_name = f"{gal4}\n(n={num_flies})"
            else:
                new_name = f"{gal4}\n(n={num_flies})"
        events.replace(gen, new_name, inplace=True)
        if len(new_order)>0:
            try:
                order_id = np.where(np.array(new_order)==gen)[0][0]
                new_order[order_id] = new_name
            except:
                pass
        if hue_order:
            try:
                order_id = np.where(np.array(hue_order)==gen)[0][0]
                hue_order[order_id] = new_name
            except:
                pass

    if horizontal:
        x_temp = x
        y_temp = y
        x = y_temp
        y = x_temp

    #plt.figure(figsize=(20, 6))

    if use_bins and split_flies:
        start_time_bins = [int(b.split("min-")[0]) for b in bins]
        ordered_start_time = np.sort(start_time_bins)
        ordered_bins = [b for s in ordered_start_time
                          for b in bins
                        if b[:len(f"{s}min")]==f"{s}min"]
        order = ordered_bins

        cat_plot = sns.catplot(x=x,
                               y=y,
                               #hue="Fly",
                               col="Bin",
                               data=events,
                               fliersize=0,
                               order=hue_order,
                               col_order=order,
                               notch=notch,
                               aspect=0.4,
                               width=0.5,
                               sharey=True,
                               kind="box",
                               ax=ax)

        for col_val, ax in cat_plot.axes_dict.items():
            for patch in ax.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.25))
        

        cat_plot.map_dataframe(sns.stripplot,
                               x=x,
                               y=y,
                               hue="Fly",
                               data=events,
                               dodge=True,
                               jitter=True,
                               zorder=0,
                               palette="Accent",
                               size=size,
                               ax=ax)

        #cat_plot.despine(left=True)
        #if hue_order:
        #    len_legend = len(hue_order)
        #else:
        #    len_legend = len(genotypes)
            
        #handles, labels = ax.get_legend_handles_labels()
        #ax.legend(handles[len_legend:], labels[len_legend:])


    elif use_bins:
        start_time_bins = [int(b.split("min-")[0]) for b in bins]
        ordered_start_time = np.sort(start_time_bins)
        ordered_bins = [b for s in ordered_start_time
                          for b in bins
                        if b[:len(f"{s}min")]==f"{s}min"]
        order = ordered_bins
        ax = sns.boxplot(x="Bin",
                         y=y,
                         hue=x,
                         data=events,
                         fliersize=0,
                         order=order,
                         hue_order=hue_order,
                         notch=notch,
                         ax=ax)
        #plt.yscale('log')
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.25))
        ax = sns.stripplot(x="Bin",
                           y=y,
                           hue=x,
                           data=events,
                           dodge=True,
                           jitter=True,
                           zorder=0,
                           size=size,
                           order=order,
                           hue_order=hue_order,
                           ax=ax)
        if hue_order:
            len_legend = len(hue_order)
        else:
            len_legend = len(genotypes)
            
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[len_legend:], labels[len_legend:])

    else:
        if len(new_order) == 0:
            new_order = None
        ax = sns.boxplot(x=x,
                         y=y,
                         data=events,
                         fliersize=0,
                         order=new_order,
                         palette=palette,
                         notch=notch)
        #plt.yscale('log')
        for patch in ax.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.4))
        ax = sns.stripplot(x=x,
                           y=y,
                           data=events,
                           color="black",
                           #palette=palette,
                           dodge=True,
                           jitter=True,
                           zorder=0,
                           size=size,
                           order=new_order)
        
        for pval, idx in p_values.items():
            if not horizontal:
                ax.axvline(idx, color='r', linestyle='--')
                ax.annotate(pval,(idx+1,1.025),fontsize=14)
            else:
                ax.axhline(idx, color='r', linestyle='--')
                ax.annotate(pval,(0.325, idx+1),fontsize=14)
        if len(ci) > 0:
            if not horizontal:
                ax.axhspan(ci[0],ci[1], color='gray', alpha=0.4)
            else:
                ax.axvspan(ci[0],ci[1], color='gray', alpha=0.4)

    
    #plt.xticks(fontsize=7, rotation=90)# Set font size for X-axis label
    if len(order) > 20 and not horizontal:
        plt.xticks(fontsize=6, rotation=90)
        plt.yticks(fontsize=20)
    elif len(order) > 20 and horizontal:
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=6)
    else:
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    
    #plt.yscale('log')
    #plt.ylim([0,1.05])
    if horizontal:
        plt.xlim([0.3,1.05])
    else:
        plt.ylim([0.3,1.05])

    if save_fig:
        plt.savefig(name_fig, bbox_inches="tight", format="pdf")
    else:
        plt.tight_layout(rect=[0.035, 0.130, 0.995, 0.995])
    
        plt.show()
