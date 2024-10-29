import numpy as np
import pandas as pd
import pickle
from Sociability_Learning.utils_connectomics import *


output_file_name = "connections_random_networks.pkl"

def main():
    nodes, edges = get_nodes_and_edges()
    df_all_lines = df_flywire_names.copy()
    num_hits = 13
    prop_hits = {"MBON":8,"MBIN":1,"MBEN":2,"LH":2}
    real_hits = [hits_learning, hits_reaction_up]

    sel_hits = {"Real hits": 1,
                "Random hits": 10,
                "Prop hits": 10
                }

    all_data = pd.DataFrame()
    for type_hits, reps in sel_hits.items():
        for i in range(reps):
            if type_hits == "Real hits":
                all_hits = real_hits
            if type_hits == "Random hits":
                all_hits = [list(df_all_lines['flywire_name'].sample(n=13).values)]
            if type_hits == "Prop hits":
                all_hits = [list(df_all_lines.loc[df_all_lines["brain_region"]==k]["flywire_name"].sample(n=num).values) for k, num in prop_hits.items()]

            hits_ids = get_ids_from_names(all_hits, nodes)
            all_connections, cell_type_connections, downstream_ids = get_names_from_downstream(hits_ids, edges, nodes)
            all_connections2, cell_type_connections2, downstream_ids2 = get_names_from_downstream(downstream_ids, edges, nodes)
            connections = find_connections(cell_type_connections, all_connections, cell_connections2=cell_type_connections2, all_connections2=all_connections2)
            direct_connections = connections.loc[connections["interneuron"].isna()]
            _, direct_connected = select_interneurons(direct_connections)
            interneurons, connected = select_interneurons(connections)
            all_neurons = np.unique(all_connections["name_pre"])
            prop_dir = len(direct_connected)#/len(all_neurons)
            prop_one = (len(connected)-len(direct_connected))#/len(all_neurons)
            prop_plus = (len(all_neurons)-len(connected))#/len(all_neurons)
            network_df = pd.DataFrame({"Network": type_hits,
                                       "Direct connections": [prop_dir],
                                       "1 hop connections": [prop_one],
                                       "2+ hops connections": [prop_plus],
                                       })
            all_data = all_data.append(network_df, ignore_index=True)

    #all_data.to_pickle(output_file_name)

    plot_analysis_random_networks(all_data)

   
if __name__ == "__main__":
    main()
