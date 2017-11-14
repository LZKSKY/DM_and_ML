import json
from Document import Document

class DocumentSet:

    def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0  # The number of documents
        # self.clusterNoArray = []
        self.documents = []
        last_id = 0
        with open(dataDir) as input:
            line = input.readline()
            while line:
                self.D += 1
                obj = json.loads(line)
                text = obj['textCleaned']
                if int(obj['tweetId']) < last_id:
                    print("ERROR: IDs are not in ascending order! ID:", int(obj['tweetId']))
                last_id = int(obj['tweetId'])
                document = Document(text, wordToIdMap, wordList, int(obj['tweetId']))
                self.documents.append(document)
                line = input.readline()
        print("number of documents is ", self.D)