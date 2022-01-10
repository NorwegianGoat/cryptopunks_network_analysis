from re import sub
from typing import List
from networkx.algorithms.core import k_core
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.smallworld import sigma
from networkx.classes.digraph import DiGraph
from networkx.classes.function import degree
from networkx.classes.graph import Graph
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.multigraph import MultiGraph
from networkx.readwrite import text
from numpy import log
import requests
import json
import time
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
from scipy import stats


def get_last_block() -> int:
    query = "https://api.etherscan.io/api?apikey=" + __API_KEY + "&module=block&action=getblocknobytime&timestamp=" + \
            str(int(time.time())) + "&closest=before"
    response = json.loads(requests.get(query).text)
    print("Latest block is: ", response['result'])
    return int(response['result'])


__API_KEY = open('./api.key', 'r').readline()
__FROM_BLOCK = 3914495  # CryptoPunks creation block
__TO_BLOCK = get_last_block()
__NULL_ADDRESS = '0x0000000000000000000000000000000000000000'
__document = None


def get_data():
    t0 = time.time()
    start = __FROM_BLOCK
    while (True):
        # Execute query
        query = "https://api.etherscan.io/api?apikey=" + __API_KEY + "&module=logs&action=getLogs&fromBlock=" + \
            str(start) + "&toBlock=latest" + \
            "&address=0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB"
        print(query, "\nREMAINING BLOCKS: ", (__TO_BLOCK - start))
        response = requests.get(query)
        # Re-execute the query in case of error
        while response.status_code != 200:
            error_string = "Request for block ", start, " failed with code " + \
                response.status_code + ". Retrying."
            print(error_string)
            response = requests.get(query)
        # Save data only in case there are events
        response = json.loads(response.text)
        to = int(response['result']
                 [-1]['blockNumber'], 16)
        if (response['status'] == '1'):
            with open("./logs_data/" + str(start) + "_" + str(to) + ".json", 'w') as writer:
                print('Writing data from block ' +
                      str(start) + " to block " + str(to))
                json.dump(response['result'], writer, indent=4)
        ''' If start equals to to it means that there aren't events in the next
        block, so we break because we have retrieved all the events. Otherwise
        we have other events to process and we set the next block to query
        equals to the last we got, this happens because etherscan limits the
        event retrieval to 1000 events and if we go to the next block we may
        loose some events. '''
        if (start == to):
            t1 = time.time()
            print('Data retrieval took:', t1 - t0, ' seconds.')
            break
        else:
            start = to


def parse_event(event: dict, i: int) -> str:
    # Event is returned in form of timestamp, event type, from, to, punkId
    event_type = event['topics'][0]
    event['timeStamp'] = str(int(event['timeStamp'], 16))
    # Assign(address indexed to, uint256 punkIndex);
    if event_type == '0x8a0e37b73a0d9c82e205d4d1a3ff3d0b57ce5f4d7bccf6bac03336dc101cb7ba':
        event['topics'][0] = "Assign"
        event['topics'][1] = "0x"+event['topics'][1][26:]
        event['data'] = str(int(event['data'], 16))
        event = event['timeStamp'] + ',' + event['topics'][0] + ',' + __NULL_ADDRESS + ',' + \
            event['topics'][1] + ',' + event['data'] + '\n'
    # PunkBought(uint indexed punkIndex, uint value, address indexed fromAddress, address indexed toAddress);
    elif event_type == '0x58e5d5a525e3b40bc15abaa38b5882678db1ee68befd2f60bafe3a7fd06db9e3':
        event['topics'][0] = "PunkBought"
        event['topics'][1] = str(int(event['topics'][1], 16))
        event['topics'][2] = "0x"+event['topics'][2][26:]
        event['topics'][3] = "0x"+event['topics'][3][26:]
        # If a punk is sold by accepting an offer the destination in the logs is the null address.
        # This if fixes it
        if (event['topics'][3] == __NULL_ADDRESS):
            event['topics'][3] = "0x"+__document[i-1]["topics"][2][26:]
        event = event['timeStamp'] + ","+event['topics'][0] + "," + \
            event['topics'][2] + "," + event['topics'][3] + \
            "," + event['topics'][1] + '\n'
    # PunkTransfer(address indexed from, address indexed to, uint256 punkIndex);
    elif event_type == '0x05af636b70da6819000c49f85b21fa82081c632069bb626f30932034099107d8':
        event['topics'][0] = "PunkTransfer"
        event['topics'][1] = "0x"+event['topics'][1][26:]
        event['topics'][2] = "0x"+event['topics'][2][26:]
        event['data'] = str(int(event['data'], 16))
        event = event['timeStamp'] + ","+event['topics'][0] + "," + event['topics'][1] + \
            "," + event['topics'][2] + "," + event['data'] + '\n'
    else:
        # Uninteresting event
        return None
    return event


def parse_and_filter_data():
    # Merge all files in a single csv containing only some specific events
    downloaded_files = os.listdir('./logs_data')
    events = ["timestamp,event_type,source,target,punk_id\n"]
    with open('./out/all_exchanges.csv', 'w') as writer:
        for file in downloaded_files:
            with open('./logs_data/' + file, 'r') as reader:
                global __document
                __document = json.load(reader)
            for i, event in enumerate(__document):
                parsed_event = parse_event(event, i)
                if(parsed_event):
                    events.append(parsed_event)
        writer.writelines(events)


def rm_duplicates():
    # Removes duplicated lines and sorts the exchanges by timestamp
    dataset = pd.read_csv("./out/all_exchanges.csv")
    print("Before duplicate cleaning:", dataset.shape[0], dataset.shape[1])
    dataset.drop_duplicates(inplace=True)
    print("After duplicate cleaning:", dataset.shape[0], dataset.shape[1])
    dataset.to_csv("./out/all_exchanges.csv", index=False, mode='w')


def data_enrichment():
    # Reads the log csv and adds punk_type to the edge list.
    # Punk data is read from data taken from
    # https://github.com/cryptopunksnotdead/punks.attributes/tree/master/original
    logs_data = pd.read_csv("./out/all_exchanges.csv")
    punk_data_files = os.listdir("./punks_data")
    punk_data_files.remove('README.md')
    punk_data = [pd.read_csv('./punks_data/' + file, skipinitialspace=True)
                 for file in punk_data_files]
    punk_data = pd.concat(punk_data)
    punk_data.sort_values(by=["id"], inplace=True)
    punk_data.to_csv("./punks_data/0-9999.csv", index=False)
    punk_data.rename(columns={"type": "punk_type",
                     "id": "punk_id"}, inplace=True)
    logs_data = logs_data.merge(punk_data.iloc[:, 0: 2], on="punk_id")
    logs_data.sort_values(by=['timestamp'], inplace=True)
    logs_data.to_csv("./out/all_exchanges.csv", index=False, mode='w')


def data_split():
    # Exchanges splitted by type
    logs_data = pd.read_csv("./out/all_exchanges.csv")
    human_data = logs_data.loc[logs_data["punk_type"] == "Human"]
    human_data.to_csv("./out/human_exchanges.csv", index=False)
    ape_data = logs_data.loc[logs_data["punk_type"] == "Ape"]
    ape_data.to_csv("./out/ape_exchanges.csv", index=False)
    zombie_data = logs_data.loc[logs_data["punk_type"] == "Zombie"]
    zombie_data.to_csv("./out/zombie_exchanges.csv", index=False)
    alien_data = logs_data.loc[logs_data["punk_type"] == "Alien"]
    alien_data.to_csv("./out/alien_exchanges.csv", index=False)


def add_rare_freq(fg: MultiDiGraph):
    logs_data = pd.read_csv("./out/all_exchanges.csv")
    logs_data = logs_data.loc[logs_data["punk_type"]
                              != "Human"].value_counts(subset="target")
    # logs_data.to_csv("./out/account_info.csv")
    logs_data = logs_data.to_dict()
    for node in fg.nodes():
        if node in logs_data:
            fg.nodes[node]["rare_freq"] = logs_data[node]
        else:
            fg.nodes[node]["rare_freq"] = 0


def add_node_rarity(bg: MultiDiGraph):
    punk_data = pd.read_csv('./punks_data/0-9999.csv')
    for node in bg.nodes():
        if node in punk_data["id"]:
            punk_type = punk_data.iloc[node]["type"]
            if punk_type == "Human":
                bg.nodes[node]["node_rarity"] = 0
            elif punk_type == "Zombie":
                bg.nodes[node]["node_rarity"] = 1
            elif punk_type == "Ape":
                bg.nodes[node]["node_rarity"] = 2
            elif punk_type == "Alien":
                bg.nodes[node]["node_rarity"] = 3
        else:
            bg.nodes[node]["node_rarity"] = -1


def graphs_creation():
    # Full matrix
    data = pd.read_csv("./out/all_exchanges.csv")
    fg: MultiDiGraph = nx.from_pandas_edgelist(
        data, edge_attr=True, create_using=nx.MultiDiGraph)
    # Add attributes to data
    add_rare_freq(fg)
    print("[FULL GRAPH]", fg)
    # Rare and common exchanges list
    common_exchanges = []
    rare_exchanges = []
    for edge in fg.edges(data=True, keys=True):
        if edge[3]["punk_type"] == "Human":
            common_exchanges.append(edge[0:3])
        else:
            rare_exchanges.append(edge[0:3])
    # Remove rare exchanges from common exchanges matrix
    cg: MultiDiGraph = fg.copy()
    cg.remove_edges_from(rare_exchanges)
    print("[COMMON GRAPH]", cg)
    # Remove common exchange from rare matrix
    rg: MultiDiGraph = fg.copy()
    rg.remove_edges_from(common_exchanges)
    print("[RARE GRAPH]", rg)
    # Bipartite graph source = Addresses, target = NFT
    data.drop(columns='source', inplace=True)
    data.rename(columns={"target": "source",
                "punk_id": "target"}, inplace=True)
    bg = nx.from_pandas_edgelist(
        data, edge_attr=True, create_using=nx.MultiDiGraph)
    add_node_rarity(bg)
    print("[BIPARTITE GRAPH]", bg)
    return fg, cg, rg, bg


def deg_distr_analysis(degrees: List, graph_name: str):
    # Pareto analysis
    # Copy values is needed because for pareto analysis we need to order the list,
    # but for correlation analysis we need the original order of the elements
    # for the plot node_rarity - degree in order to have data matching
    degrees = degrees[:]
    degrees.sort()
    degrees.reverse()
    degree_sum = sum(degrees)
    total_nodes = len(degrees)
    sub_total = 0
    for i in range(0, total_nodes):
        sub_total += degrees[i]
        percentage = (sub_total/degree_sum)*100
        if percentage >= 80:
            print((i/total_nodes)*100, "% nodes are involved in ",
                  percentage, "% of exchanges")
            break
    # Plot degree distr
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False)
    ax1.set_yscale("log")
    ax1.hist(degrees, bins=30)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.hist(degrees, bins="auto")
    fig.supxlabel('Degree')
    fig.supylabel('Frequency')
    fig.savefig('./out/'+graph_name+"_deg_dist")
    plt.clf()
    # Scale free analysis
    fit = powerlaw.Fit(degrees, discrete=True, xmin=1)
    print("Power law alpha: ", fit.alpha,
          " Power law xmin: ", fit.xmin)
    axis = fit.plot_ccdf()
    fit.power_law.plot_ccdf(ax=axis, color='r', linestyle='--')
    fit.lognormal.plot_ccdf(ax=axis, color='g', linestyle='--')
    fit.stretched_exponential.plot_ccdf(ax=axis, color='b', linestyle='--')
    plt.legend(['Observed network', 'Power law',
               'Lognormal', 'Stretched exp.'])
    plt.xlabel("Degree")
    plt.ylabel("CDF")
    plt.savefig('./out/'+graph_name+'_ccdf')
    plt.clf()
    power_law_lognormal = fit.distribution_compare(
        "power_law", "lognormal", normalized_ratio=True)
    power_law_stretched = fit.distribution_compare(
        "power_law", "stretched_exponential", normalized_ratio=True)
    log_normal_stretched = fit.distribution_compare(
        "lognormal", "stretched_exponential", normalized_ratio=True)
    print("Power law vs lognormal: ", power_law_lognormal,
          " Power law vs stretched exponential", power_law_stretched, " Lognormal vs stretched exponential ", log_normal_stretched)


def graph_analysis(mdg: MultiDiGraph, graph_name: str):
    # graph name is used while saving images and for analysis
    mg: MultiGraph = mdg.to_undirected()
    # Null address removal. We will study the network using also non human addresses
    # mg.remove_node(__NULL_ADDRESS)
    # mg.remove_nodes_from(list(nx.isolates(mg)))
    g = nx.Graph(mdg)
    g.remove_edges_from(nx.selfloop_edges(g))
    if graph_name != "bg":
        # Degree analysis
        degrees = [degree for node, degree in mg.degree()]
        deg_distr_analysis(degrees, graph_name)
        # Average path lenght
        avg_path = nx.algorithms.average_shortest_path_length(mg)
        print("Average path length: ", avg_path)
        # Diameter
        diameter = nx.algorithms.diameter(mg)
        print("Diameter: ", diameter)
        # Clustering coefficent
        c_coefficent = nx.algorithms.average_clustering(g)
        print("Clustering coefficent: ", c_coefficent)
        # k-cores analysis
        n_nodes = range(1, 15)
        k_cores = [nx.algorithms.k_core(
            g, k).order() for k in n_nodes]
        print("K-cores analysis: ", k_cores)
        plt.plot(n_nodes, k_cores, marker="s")
        plt.xlabel("K")
        plt.ylabel("Nodes")
        plt.savefig('./out/'+graph_name+"_k_cores")
        # Homophily
        degree_assort = nx.algorithms.degree_assortativity_coefficient(mg)
        rarity_assort = nx.algorithms.numeric_assortativity_coefficient(
            mg, attribute="rare_freq")
        print("Homophily degree: ", degree_assort,
              " Homophily rarity: ", rarity_assort)
    else:
        degrees = [degree for node, degree in mg.degree()
                   if node in range(0, 10001)]
        deg_distr_analysis(degrees, graph_name)
        node_rarity = [values["node_rarity"] for id, values in mg.nodes(
            data=True) if id in range(0, 10001)]
        plt.scatter(node_rarity, degrees)
        plt.xlabel("node_rarity")
        plt.ylabel("degree")
        plt.savefig('./out/'+graph_name+"_rarity_deg")
        r_coefficent = stats.pearsonr(node_rarity, degrees)
        print("Pearson coefficent. r: ",
              r_coefficent[0], " p: ", r_coefficent[1])


def debug(data: MultiDiGraph):
    data = nx.Graph(data)
    sigma = nx.algorithms.sigma(data)
    print("Sigma: ", sigma)
    omega = nx.algorithms.omega(data)
    print("Omega: ", omega)


if __name__ == "__main__":
    # get_data()
    # parse_and_filter_data()
    # rm_duplicates()
    # data_enrichment()
    # data_split()
    fg, cg, rg, bg = graphs_creation()
    '''edgelists of common exchanges and rare exchanges. Saved for further
    visual inspection with gephi. Conceptually rare_exchanges is the sum of
    ape_exchanges, alien_exchanges and zombie_echanges.
    common_exchanges is all_exchanges - rare_exchanges'''
    # nx.to_pandas_edgelist(cg).to_csv(
    #    "./out/common_exchanges.csv", index=False, mode="w")
    # nx.to_pandas_edgelist(rg).to_csv(
    #    "./out/rare_exchanges.csv", index=False, mode="w")
    '''Edgelist of the bipartite graph Nft - owner'''
    # nx.to_pandas_edgelist(bg).to_csv(
    #    ./out/bipartite.csv", index=False, mode="w")
    graph_analysis(fg, "fg")
    graph_analysis(bg, "bg")
    # debug(fg)
    pass
