from DocumentSet import DocumentSet
from Model import Model

class MStream:
    dataDir = "data/"
    outputPath = "result/"

    def __init__(self, K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
        self.K = K
        self.KIncrement = KIncrement
        self.alpha = alpha
        self.beta = beta
        self.iterNum = iterNum
        self.dataset = dataset
        self.timefil = timefil
        self.sampleNum = sampleNum
        self.wordsInTopicNum = wordsInTopicNum
        self.wordList = []
        self.wordToIdMap = {}

    def getDocuments(self):
        self.documentSet = DocumentSet(self.dataDir + self.dataset, self.wordToIdMap, self.wordList)
        self.V = self.wordToIdMap.__len__()

    def runMStream(self, sampleNo):
        ParametersStr = "K" + str(self.K) + "iterNum" + str(self.iterNum) + \
                        "SampleNum" + str(self.sampleNum) + "alpha" + str(round(self.alpha, 3)) + \
                        "beta" + str(round(self.beta, 3))
        model = Model(self.K, self.KIncrement, self.V, self.iterNum, self.alpha, self.beta, self.dataset,
                      ParametersStr, sampleNo, self.wordsInTopicNum, self.dataDir + self.timefil)
        model.run(self.documentSet, self.outputPath, self.wordList)

    def runMStreamIter(self, sampleNo):
        ParametersStr = ""
        model = Model(self.K, self.KIncrement, self.V, self.iterNum, self.alpha, self.beta, self.dataset,
                      ParametersStr, sampleNo, self.wordsInTopicNum)
        model.intialize(self.documentSet)
        model.gibbsSamplingIter(self.documentSet, self.iterNum, self.sampleNum, self.outputPath, self.wordList)