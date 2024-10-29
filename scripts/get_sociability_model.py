from Sociability_Learning.utils_connectomics import *
   
def main():
    nodes, edges = get_nodes_and_edges()
    all_hits = [hits_learning, hits_reaction_up, hits_reaction_down]
    hits_ids = get_ids_from_names(all_hits, nodes)
    all_connections, cell_type_connections, downstream_ids = get_names_from_downstream(hits_ids, edges, nodes)
    all_connections2, cell_type_connections2, downstream_ids2 = get_names_from_downstream(downstream_ids, edges, nodes)
    connections = find_connections(cell_type_connections, all_connections, cell_connections2=cell_type_connections2, all_connections2=all_connections2)
    direct_connections = connections.loc[connections["interneuron"].isna()]
    _, directly_connected = select_interneurons(direct_connections)
    connections_selected, all_connected = select_interneurons(connections)
    
    print(connections_selected)

   
if __name__ == "__main__":
    main()

