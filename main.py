from web3 import Web3
import requests
import json

__ENDPOINT_URI = "https://cloudflare-eth.com/"
__API_KEY = open('./api.key','r').readline()
__FROM_BLOCK = 3914495
__TO_BLOCK = 13422401
__BLOCK_STEP = 10


def get_data():
    __from = __FROM_BLOCK
    for __to in range(__FROM_BLOCK, __TO_BLOCK, __BLOCK_STEP):
        # Execute query
        query = "https://api.etherscan.io/api?apikey=" + __API_KEY + "&module=logs&action=getLogs&fromBlock=" + \
            str(__from) + "&toBlock=" + str(__to) + \
            "&address=0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB"
        print(query, "\n REMAINING REQUESTS: ", (__TO_BLOCK - __from)/__BLOCK_STEP)
        response = requests.get(query)
        # Log error and re-execute the query
        while response.status_code != 200:
            error_string = "Request for block ", __from, " failed with code " + \
                response.status_code + ". Retrying."
            print(error_string)
            response = requests.get(query)
        response = json.loads(response.text)
        # Save data only in case there are events
        if (response['status'] == '1'):
            with open("./data/" + str(__from) + "_" + str(__to) + ".json", 'w') as writer:
                print('Writing data from block ' +
                      str(__from) + " to block " + str(__to))
                json.dump(response['result'], writer, indent=4)
        __from = __to + 1

def clean_data(w3: Web3):
    pass


if __name__ == "__main__":
    get_data()
