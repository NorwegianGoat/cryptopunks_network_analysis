from networkx.classes.multidigraph import MultiDiGraph
import requests
import json
import time
import os
import pandas as pd
import networkx as nx
import numpy


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
    logs_data = logs_data.merge(punk_data.iloc[:, 0: 2].rename(
        columns={"id": "punk_id", "type": "punk_type"}), on="punk_id")
    punk_data.rename(columns={"type": "punk_type"}, inplace=True)
    logs_data.sort_values(by=['timestamp'], inplace=True)
    logs_data.to_csv("./out/all_exchanges.csv", index=False, mode='w')


def data_split():
    # Exchanges split by type
    logs_data = pd.read_csv("./out/all_exchanges.csv")
    human_data = logs_data.loc[logs_data["punk_type"] == "Human"]
    human_data.to_csv("./out/human_exchanges.csv", index=False)
    ape_data = logs_data.loc[logs_data["punk_type"] == "Ape"]
    ape_data.to_csv("./out/ape_exchanges.csv", index=False)
    zombie_data = logs_data.loc[logs_data["punk_type"] == "Zombie"]
    zombie_data.to_csv("./out/zombie_exchanges.csv", index=False)
    alien_data = logs_data.loc[logs_data["punk_type"] == "Alien"]
    alien_data.to_csv("./out/alien_exchanges.csv", index=False)


def matrices_creation():
    # Full matrix
    fg: MultiDiGraph = nx.from_pandas_edgelist(
        pd.read_csv("./out/all_exchanges.csv"), edge_attr=True, create_using=nx.MultiDiGraph)
    print("[FULL MATTRIX]", fg)
    # Removal of human exchanges from ""
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
    print("[COMMON MATRIX]", cg)
    # Remove common exchange from rare matrix
    rg: MultiDiGraph = fg.copy()
    rg.remove_edges_from(common_exchanges)
    print("[RARE MATRIX]", rg)
    return fg, cg, rg


def graph_analysis(matrix):
    pass


if __name__ == "__main__":
    # get_data()
    # parse_and_filter_data()
    # rm_duplicates()
    # data_enrichment()
    # data_split()
    fg, cg, rg = matrices_creation()
    '''edgelists of common exchanges and rare exchanges. Saved for further
    visual inspection with gephi. Conceptually rare_exchanges is the sum of
    ape_exchanges, alien_exchanges and zombie_echanges. 
    common_exchanges is all_exchanges - rare_exchanges'''
    nx.to_pandas_edgelist(cg).to_csv(
        "./out/common_exchanges.csv", index=False, mode="w")
    nx.to_pandas_edgelist(rg).to_csv(
        "./out/rare_exchanges.csv", index=False, mode="w")
    t = nx.from_numpy_matrix(numpy.dot(nx.adjacency_matrix(
        nx.Graph(rg)), nx.adjacency_matrix(nx.Graph(cg))))
    pass
