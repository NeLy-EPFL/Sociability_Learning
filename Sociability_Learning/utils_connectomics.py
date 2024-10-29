import os
import pkgutil
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from pathlib import Path


hits_learning= ['LHPV4a9',
                'MBON22',
                'MBON19',
                'MBON04',
                #'ORCO-GR63a-mutant',
                'PAM12',
                #'LHIN-PLP-PN1',
                #'PLP156',
                #'KCapbp-m',
                ]

hits_reaction_up = [#'All-PAMs',
                    #'GH146',
                    'KCapbp-m',
                    "KCa'b'-m",
                    #'LHIN-PLP-PN1',
                    'PLP156',
                    #'LO-PN2',
                    #'SLP448',
                    'MBON03',
                    'MBON11',
                    'MBON12',
                    'MBON07',
                    'MBON09',
                    'PAM10']

hits_reaction_down = [#'PLP-PN2',
                      #'SLP362']#'LHPV5a1'
                     ]

package = pkgutil.get_loader("Sociability_Learning")
path_package = Path(package.get_filename())
flywire_names = os.path.join(path_package.parents[1],
                             'data',
                             'flywire_data',
                             'all_lines_flywire_names.csv')

df_flywire_names = pd.read_csv(flywire_names, header=0)
not_hits = [line["flywire_name"] for index, line in df_flywire_names.iterrows() if line["flywire_name"] not in hits_learning and line["flywire_name"] not in hits_reaction_up and line["flywire_name"] not in hits_reaction_down]

def get_nodes_and_edges(add_hemibrain=True):
    # Read Codex dump
    neurons_path = os.path.join(path_package.parents[1],
                                'data',
                                'flywire_data',
                                'neurons.csv')
    classification_path = os.path.join(path_package.parents[1],
                                       'data',
                                       'flywire_data',
                                       'classification.csv')
    labels_path = os.path.join(path_package.parents[1],
                               'data',
                               'flywire_data',
                               'labels.csv')
    connections_path = os.path.join(path_package.parents[1],
                                    'data',
                                    'flywire_data',
                                    'connections.csv')
    neurons = pd.read_csv(neurons_path)
    classification = pd.read_csv(classification_path)
    labels = pd.read_csv(labels_path)
    connections = pd.read_csv(connections_path)

    node_info = neurons.copy()
    node_info = pd.merge(node_info, classification, on="root_id", how="left")
    node_info = pd.merge(node_info, labels, on="root_id", how="left")

    node_attr_names = [
    "group",
    "nt_type",
    "nt_type_score",
    "flow",
    "super_class",
    "class",
    "sub_class",
    "cell_type",
    "hemibrain_type",
    "hemilineage",
    "side",
    "nerve",
    "label",
    ]
    node_info_sel = node_info[["root_id", *node_attr_names]].drop_duplicates(subset="root_id").reset_index()#.set_index("root_id")

    edge_info = connections.copy()
    edge_info = pd.merge(
        edge_info,
        node_info[["root_id", "nt_type"]],
        left_on="pre_root_id",
        right_on="root_id",
        how="left",
        suffixes=("", "_unified"),
    )
    edge_info["nt_type"] = np.where(
        edge_info["nt_type_unified"].notnull(),
        edge_info["nt_type_unified"],
        edge_info["nt_type"],
    )
    edge_info = edge_info.drop(columns=["root_id", "nt_type_unified"])

    node_info_sel["dataset"] = ["flywire"]*len(node_info_sel)
    edge_info["dataset"] = ["flywire"]*len(edge_info)
    if add_hemibrain:
        hemibrain_nodes, hemibrain_edges = get_nodes_and_edges_hemibrain()
        node_info_sel = pd.concat([node_info_sel, hemibrain_nodes[["root_id","hemibrain_type","dataset"]]], ignore_index=True)
        edge_info = pd.concat([edge_info, hemibrain_edges], ignore_index=True)
    
    return node_info_sel, edge_info

def get_nodes_and_edges_hemibrain():
    neurons_path = os.path.join(path_package.parents[1],
                                'data',
                                'hemibrain-v1_2',
                                'traced-neurons.csv')
    connections_path = os.path.join(path_package.parents[1],
                                'data',
                                'hemibrain-v1_2',
                                'traced-total-connections.csv')
    neurons = pd.read_csv(neurons_path)
    connections = pd.read_csv(connections_path)

    node_info = neurons.copy()
    node_info.columns=["root_id","hemibrain_type","instance"]
    node_info["dataset"]=["hemibrain"]*len(node_info)
    
    edge_info = connections.copy()
    edge_info.columns = ["pre_root_id", "post_root_id", "syn_count"]
    edge_info["dataset"]=["hemibrain"]*len(edge_info)
    
    return node_info, edge_info

def check_name(name, id_):
    if not isinstance(name, str):
        name = f"no_name_{id_}"
    if "'" in name:
        name = name.replace("'","p")
    if "AD1a1" in name or "AD1f1" in name:
        name = "LHAD1a1/f1"
    if "PV5g1" in name or "PV5g2" in name:
        name = "LHPV5g1/g2"
    return name

def get_ids_from_names(all_names, nodes):
    all_ids = {}
    for name_list in all_names:
        for name in name_list:
            list_ids = nodes.loc[(nodes["cell_type"]==name) &
                                 (nodes["dataset"]=="flywire")]["root_id"].to_list()

            if len(list_ids)==0:
                #if len(name) > 3:
                #    list_ids = nodes.loc[(nodes["hemibrain_type"].str.contains(name).fillna(False)) &
                #                         (nodes["dataset"]=="flywire")]["root_id"].to_list()
                #else:
                list_ids = nodes.loc[(nodes["hemibrain_type"]==name) &
                                     (nodes["dataset"]=="flywire")]["root_id"].to_list()
            if len(list_ids)==0:
                list_ids = nodes.loc[(nodes["label"].str.contains(name).fillna(False))&
                                     (nodes["dataset"]=="flywire")]["root_id"].to_list()

            if len(name) > 3:
                ids_hemibrain = nodes.loc[(nodes["hemibrain_type"].str.contains(name).fillna(False))&
                                          (nodes["dataset"]=="hemibrain")]["root_id"].to_list()
            else:
                ids_hemibrain = nodes.loc[(nodes["hemibrain_type"]==name)&
                                          (nodes["dataset"]=="hemibrain")]["root_id"].to_list()
            list_ids.extend(ids_hemibrain)
            
            if len(list_ids)==0:
                print(f"{name} not found")

            name = check_name(name, 0)
            
            if name in all_ids.keys():
                all_ids[name].extend(list_ids)
            else:
                all_ids[name] = list_ids
    return all_ids

def get_names_from_ids(all_ids, nodes):
    if not isinstance(all_ids, list):
        all_ids = [all_ids]
    all_names={}
    for down_id in all_ids:
        name_down = nodes.loc[nodes["root_id"]==down_id]["hemibrain_type"].values[0]
        if not isinstance(name_down, str):
            name_down = nodes.loc[nodes["root_id"]==down_id]["label"].values[0]
        if not isinstance(name_down, str):
            name_down = nodes.loc[nodes["root_id"]==down_id]["cell_type"].values[0]
        else:
            name_down = name_down.split(";")[0]

        name_down = check_name(name_down, down_id)

        if name_down not in all_names.keys():
            all_names[down_id] = name_down
            
    return all_names

def get_names_from_downstream(dict_names, edges, nodes, min_syn=10, print_info=False):
    all_connections = pd.DataFrame()
    cell_type_connections = {}
    downstream_ids = {}
    print("Computing connections")
    for idx, (name, ids) in enumerate(dict_names.items()):
        if print_info:
            print(f"Computing {name} ({idx+1}/{len(dict_names)})")
        if len(ids)==0:
            continue
        for i in ids:
            connections_df = edges.loc[edges["pre_root_id"]==i]
            all_downstream = connections_df.groupby(['pre_root_id', 'post_root_id']).agg({'syn_count': 'sum', 'nt_type': 'first'}).reset_index()
            sel_downstream = all_downstream.loc[all_downstream["syn_count"]>min_syn][["pre_root_id", "post_root_id", "syn_count", "nt_type"]]
            list_downstream = sel_downstream["post_root_id"].to_list()
            names_downstream = []
            for down_id in list_downstream:
                if len(all_connections)>0 and down_id in all_connections["post_root_id"].values:
                    name_down = all_connections.loc[all_connections["post_root_id"]==down_id, "name_post"].values[0]
                else:
                    name_dict = get_names_from_ids(down_id, nodes)
                    name_down = name_dict[down_id]
                    
                    if name_down in downstream_ids.keys():
                        downstream_ids[name_down].append(down_id)
                    else:
                        if name_down not in dict_names.keys():
                            downstream_ids[name_down]=[]
                            downstream_ids[name_down].append(down_id)

                names_downstream.append(name_down)

            sel_downstream["name_pre"] = [name]*len(sel_downstream)
            sel_downstream["name_post"] = names_downstream
            all_connections = all_connections.append(sel_downstream,ignore_index=True)
        cell_type_connections[name] = all_connections.loc[all_connections["name_pre"]==name]["name_post"].unique()
    return all_connections, cell_type_connections, downstream_ids

def find_connections(cell_type_connections, all_connections, cell_connections2=None, all_connections2=None, simplified=False, draw=False, print_info=False):
    G = nx.DiGraph()
    hits = list(cell_type_connections.keys())
    first_order_connections=[]
    print("Finding graph")
    for node, neighbors in cell_type_connections.items():
        G.add_node(node)
        if print_info:
            print(node)
        for neighbor in neighbors:
            if neighbor in hits and neighbor != node:
                nt_type = all_connections.loc[(all_connections["name_pre"]==node) &
                                              (all_connections["name_post"]==neighbor)]["nt_type"].unique()
                syn_count = all_connections.loc[(all_connections["name_pre"]==node) &
                                              (all_connections["name_post"]==neighbor)]["syn_count"].sum()
                
                if print_info:
                    print(node, neighbor, nt_type, syn_count)
                G.add_edge(node, neighbor)
                first_order_connections.append([node, neighbor, nt_type, syn_count])
    all_connections_df = pd.DataFrame(first_order_connections,columns=["node1","node2","nt_type1","syn_count1"])

    parent_nodes = all_connections_df["node1"].unique()

    child_nodes = all_connections_df["node2"].unique()
    
    if cell_connections2:
        second_order_connections = []
        for node, neighbors in cell_type_connections.items():
            if True:#node not in parent_nodes or node not in child_nodes:
                for neighbor in neighbors:
                    #if isinstance(neighbor, str) and not G.has_edge(node, neighbor) and neighbor not in not_hits:
                    if isinstance(neighbor, str) and neighbor in cell_connections2.keys() and not G.has_edge(node, neighbor) and neighbor not in not_hits:
                        for neighbor2 in cell_connections2[neighbor]:
                            if neighbor2 in hits and neighbor2 != node and not G.has_edge(node, neighbor2) and (neighbor2 not in parent_nodes or neighbor2 not in child_nodes):
                                nt_type1 = all_connections.loc[(all_connections["name_pre"]==node) &
                                                  (all_connections["name_post"]==neighbor)]["nt_type"].unique()
                                nt_type2 = all_connections2.loc[(all_connections2["name_pre"]==neighbor) &
                                                  (all_connections2["name_post"]==neighbor2)]["nt_type"].unique()
                                syn_count1 = all_connections.loc[(all_connections["name_pre"]==node) &
                                              (all_connections["name_post"]==neighbor)]["syn_count"].sum()
                                syn_count2 = all_connections2.loc[(all_connections2["name_pre"]==neighbor) &
                                                  (all_connections2["name_post"]==neighbor2)]["syn_count"].sum()
                                second_order_connections.append([node, neighbor, neighbor2, nt_type1, nt_type2, syn_count1, syn_count2])
                                if not simplified and print_info:
                                    print(node, neighbor, neighbor2, nt_type1, nt_type2, syn_count1, syn_count2)
                            
            else:
                print(f"{node} already has inputs and outputs")

        second_order_df = pd.DataFrame(second_order_connections,columns=["node1","interneuron","node2","nt_type1","nt_type2","syn_count1","syn_count2"])
        all_connections_df = pd.concat([all_connections_df, second_order_df], ignore_index=True)
        if simplified:
            nodes_to_connect = second_order_df["node1"].unique()
            for i, n in enumerate(nodes_to_connect):
                targets = second_order_df.loc[second_order_df["node1"]==n]["node2"].value_counts()
                interneuron = f"IN{i+1}"
                for t, count in zip(targets.index, targets):
                    if count >= np.floor(0.1*sum(targets)):
                        print(n, interneuron, t)
                        G.add_node(interneuron)
                        G.add_edge(n, interneuron)
                        G.add_edge(interneuron, t)
        else:
            for node, neighbor, neighbor2, _, _, _, _ in second_order_connections:
                if not G.has_node(neighbor):
                    G.add_node(neighbor)
                if not G.has_edge(node, neighbor):
                    G.add_edge(node, neighbor)
                G.add_edge(neighbor, neighbor2)


    if draw:
        # Draw the graph (optional)
        pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold', arrowsize=15, node_size=500, node_color='lightgray')

        hits_reaction_up_clean = [name for name in hits_reaction_up if "'" not in name]
        hits_reaction_down_clean = [name for name in hits_reaction_down if "'" not in name]
        hits_learning_clean = [name for name in hits_learning if "'" not in name]

        nx.draw_networkx_nodes(G, pos, nodelist=hits_reaction_down_clean, node_color='cyan', node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=hits_reaction_up_clean, node_color='violet', node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=hits_learning_clean, node_color='orange', node_size=500)


        # Use nx.draw_networkx to include self-loops
        #nx.draw_networkx(G, pos, with_labels=True, font_weight='bold', arrowsize=20)

        # Manually add self-loops
        #for node in G.nodes():
        #    if G.has_edge(node, node):
        #        plt.arrow(pos[node][0], pos[node][1], 0.0, 0.0, head_width=0.05, head_length=0.1, fc='r', ec='r', lw=2)

        plt.show()
    return all_connections_df

def select_interneurons(connections, count="syn_count_tot"):
    selected_interneurons = pd.DataFrame()
    connections["syn_count_tot"]=connections["syn_count1"]+connections["syn_count2"]

    interneurons_sum = connections.groupby('interneuron')[count].sum()
    interneurons_df = interneurons_sum.reset_index()
    interneurons_sorted = interneurons_df.sort_values(by=count, ascending=False)

    direct_connections = connections.loc[connections['interneuron'].isna()]
    #indirect_connections = all_connections.loc[(all_connections["Trial"]==i) & (all_connections['interneuron'].notna())]

    all_parents = direct_connections["node1"].to_list() 
    all_children = direct_connections["node2"].to_list()

    parent_nodes = list(np.unique(all_parents))
    child_nodes = list(np.unique(all_children))
    
    selected_interneurons = selected_interneurons.append(direct_connections, ignore_index=True)

    #print(parent_nodes)
    #print(child_nodes)
    for interneuron in interneurons_sorted["interneuron"].to_list():
        potential_nodes_df = connections.loc[connections["interneuron"]==interneuron]
        all_pot_parents = potential_nodes_df["node1"].to_list()
        all_pot_children = potential_nodes_df["node2"].to_list()
        
        potential_parents = np.unique(all_pot_parents)
        potential_children = np.unique(all_pot_children)
        
        not_connected_parents= [node for node in potential_parents if node not in parent_nodes]
        not_connected_children= [node for node in potential_children if node not in child_nodes]

        #print(interneuron, not_connected_parents, not_connected_children)
        
        if not_connected_parents or not_connected_children:
            selected_interneurons = selected_interneurons.append(potential_nodes_df, ignore_index=True)
            if not_connected_parents:
                parent_nodes += not_connected_parents
            
            if not_connected_children:
                child_nodes += not_connected_children

    all_connected = np.unique(parent_nodes + child_nodes)
    
    return selected_interneurons, all_connected

def get_ioh(connections):
    unique_origin = connections['node1'].value_counts()
    unique_dest = connections['node2'].value_counts()

    inputs = unique_origin.index.difference(unique_dest.index)
    outputs = unique_dest.index.difference(unique_origin.index)
    hub = unique_origin.index.intersection(unique_dest.index)

    return inputs.to_list(), outputs.to_list(), hub.to_list()

def get_percentage_inputs(cell_type, syn_count, edges, hits_ids):

    tot_sum = 0
    for id_num in hits_ids[cell_type]:
        if id_num>1000000000:
            sum_syn_count = edges.loc[(edges["post_root_id"]==id_num)&(edges["syn_count"]>10)]["syn_count"].sum()
            tot_sum += sum_syn_count

    return syn_count/tot_sum


def plot_analysis_random_networks(all_data):

    #all_data = pd.read_pickle("240913_connections_random_networks.pkl")
    
    df_melted = all_data.melt(id_vars=['Network'],
                              value_vars=['Direct connections', '1 hop connections', '2+ hops connections'],
                              var_name='Connection type',
                              value_name='Value')

    fig, ax = plt.subplots()
    ax = sns.boxplot(data=df_melted,
                     x="Connection type",
                     y="Value",
                     hue="Network",
                     fliersize=0,
                     notch=False,
                     ax=ax)
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.25))

    ax = sns.stripplot(data=df_melted,
                       x="Connection type",
                       y="Value",
                       hue="Network",
                       dodge=True,
                       jitter=True,
                       zorder=0,
                       ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    len_legend = int(len(handles)/2)
    ax.legend(handles[len_legend:], labels[len_legend:])
    #name_fig = "/mnt/upramdya_files/LOBATO_RIOS_Victor/imgs_for_paper/network_analysis.pdf"
    #plt.savefig(name_fig, bbox_inches="tight", format="pdf")
    plt.show()
        

