import requests
import json
import time
import os


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


# Event is returned in form of timestamp, event type, from, to, punkId
def parse_event(event: dict) -> str:
    event_type = event['topics'][0]
    event['timeStamp'] = str(int(event['timeStamp'], 16))
    # Assign(address indexed to, uint256 punkIndex);
    if event_type == '0x8a0e37b73a0d9c82e205d4d1a3ff3d0b57ce5f4d7bccf6bac03336dc101cb7ba':
        event['topics'][0] = "Assign"
        event['data'] = str(int(event['data'], 16))
        event = event['timeStamp'] + ',' + event['topics'][0] + ',' + __NULL_ADDRESS + ',' + \
            event['topics'][1] + ',' + event['data'] + '\n'
    # PunkBought(uint indexed punkIndex, uint value, address indexed fromAddress, address indexed toAddress);
    elif event_type == '0x58e5d5a525e3b40bc15abaa38b5882678db1ee68befd2f60bafe3a7fd06db9e3':
        event['topics'][0] = "Punk Bought"
        event['topics'][1] = str(int(event['topics'][1], 16))
        event = event['timeStamp'] + ","+event['topics'][0] + "," + event['topics'][2] + \
            "," + event['topics'][3] + "," + event['topics'][1] + '\n'
    # PunkTransfer(address indexed from, address indexed to, uint256 punkIndex);
    elif event_type == '0x05af636b70da6819000c49f85b21fa82081c632069bb626f30932034099107d8':
        event['topics'][0] = "PunkTransfer"
        event['data'] = str(int(event['data'], 16))
        event = event['timeStamp'] + ","+event['topics'][0] + "," + event['topics'][1] + \
            "," + event['topics'][2] + "," + event['data'] + '\n'
    else:
        # Uninteresting event
        return None
    return event


def filter_data():
    # Merge all files in a single csv
    downloaded_files = os.listdir('./logs_data')
    events = []
    with open('./logs_data/logs.csv', 'w') as writer:
        writer.write('timestamp, event_type, source, target, punk_id\n')
        for file in downloaded_files:
            with open('./logs_data/' + file, 'r') as reader:
                document = json.load(reader)
            for event in document:
                parsed_event = parse_event(event)
                if(parsed_event):
                    events.append(parsed_event)
        writer.writelines(events)


if __name__ == "__main__":
    # get_data()
    filter_data()
