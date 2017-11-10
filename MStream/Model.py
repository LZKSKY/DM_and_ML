import random
import sys
import os
import time
import json

class Model:
    KIncrement = 100
    smallDouble = 1e-150
    largeDouble = 1e150

    def __init__(self, K, KIncrement, V, iterNum,alpha, beta, dataset, ParametersStr, sampleNo, wordsInTopicNum):
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.Kin = K
        self.V = V
        self.iterNum = iterNum
        self.beta0 = V * beta
        self.KIncrement = KIncrement
        self.Kmax = max(K, KIncrement)
        self.sampleNo = sampleNo
        self.wordsInTopicNum = wordsInTopicNum
        self.phi_zv = []

    def intialize(self, documentSet):
        self.D = documentSet.D # The whole number of documents
        self.z = [0]*self.D # Cluster assignments of each document
        self.m_z = [0]*self.Kmax # The number of documents in cluster z
        self.n_z = [0]*self.Kmax # The number of words in cluster z
        self.n_zv = [[0]*(self.V+1)]*self.Kmax # The number of occurrences of word v in cluster z
        self.kToClusterPos = [0]*self.Kmax
        self.clusterPosTok = [0]*self.Kmax
        for k in range(self.Kmax):
            self.kToClusterPos[k] = k
            self.clusterPosTok[k] = k
        for d in range(self.D):
            document = documentSet.documents[d]
            cluster = self.sampleCluster(d, document) # Get initial cluster of each document
            self.z[d] = cluster
            self.m_z[cluster] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                self.n_zv[cluster][wordNo] += wordFre
                self.n_z[cluster] += wordFre

    def gibbsSampling(self, documentSet):

        for i in range(self.iterNum):
            print("\titer is ", i)
            for d in range(self.D):
                document = documentSet.documents[d]
                cluster = self.z[d]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)
                cluster = self.sampleCluster(d, document)
                self.z[d] = cluster
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre

    def gibbsOneIteration(self, documentSet):
        for d in range(self.D):
            document = documentSet.documents[d]
            cluster = self.z[d]
            self.m_z[cluster] -= 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                self.n_zv[cluster][wordNo] -= wordFre
                self.n_z[cluster] -= wordFre
            self.checkEmpty(cluster)
            cluster = self.sampleCluster(d, document)
            self.z[d] = cluster
            self.m_z[cluster] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                self.n_zv[cluster][wordNo] -= wordFre
                self.n_z[cluster] -= wordFre

    def sampleCluster(self, d, document):
        prob = [float(0.0)] * (self.K + 1)
        overflowCount = [0] * (self.K + 1)
        for k in range(self.K):
            cluster = self.kToClusterPos[k]
            prob[k] = self.m_z[cluster] / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if valueOfRule2 < self.smallDouble:
                        overflowCount[k] -= 1
                        valueOfRule2 *= self.largeDouble
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[k] *= valueOfRule2
        prob[self.K] = self.alpha / (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                if valueOfRule2 < self.smallDouble:
                    print(" - valueOfRule2 < self.smallDouble")
                    overflowCount[self.K] -= 1
                    valueOfRule2 *= self.largeDouble
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2
        # self.reComputeProbs(prob, overflowCount, self.K + 1)
        for k in range(1, self.K+1):
            prob[k] += prob[k-1]
        thred = random.random() * prob[self.K]
        kChoosed = 0
        while kChoosed < self.K+1:
            if thred < prob[kChoosed]:
                break
            kChoosed += 1
        if kChoosed == self.K:
            self.K += 1
            if self.K > self.Kmax:
                self.enlargeCapacity()
        return self.kToClusterPos[kChoosed]

    def checkEmpty(self, cluster):
        if self.m_z[cluster] == 0:
            k = self.clusterPosTok[cluster]
            lastClusterPos = self.kToClusterPos[self.K - 1]
            self.kToClusterPos[self.K - 1] = cluster
            self.kToClusterPos[k] = lastClusterPos
            self.clusterPosTok[cluster] = self.K - 1
            self.clusterPosTok[lastClusterPos] = k
            self.K -= 1

    def enlargeCapacity(self):
        Kmax_old = self.Kmax
        self.Kmax += self.KIncrement
        m_z_new = [self.Kmax]
        n_z_new = [self.Kmax]
        n_zv_new = [self.Kmax][self.V]
        kToClusterPos_new = [self.Kmax]
        clusterPosTok_new = [self.Kmax]
        for k in range(Kmax_old):
            m_z_new[k] = self.m_z[k]
            n_z_new[k] = self.n_z[k]
            kToClusterPos_new[k] = self.kToClusterPos[k]
            clusterPosTok_new[k] = self.clusterPosTok[k]
            for t in range(self.V):
                n_zv_new[k][t] = self.n_zv[k][t]
        for k in range(Kmax_old, self.Kmax):
            kToClusterPos_new[k] = k
            clusterPosTok_new[k] = k
        self.m_z = m_z_new
        self.n_z = n_z_new
        self.n_zv = n_zv_new
        self.kToClusterPos = kToClusterPos_new
        self.clusterPosTok = clusterPosTok_new

    def reComputeProbs(self, prob, overflowCount, K):
        max = -sys.maxsize
        for i in range(K):
            if overflowCount[i] > max and prob[i] > 0:
                max = overflowCount[i]
        for i in range(K):
            if prob[i] > 0:
                prob[i] = prob[i] * (self.largeDouble ** (overflowCount[i] - max))

    def output(self, documentSet, outputPath, wordList):
        outputDir = outputPath + self.dataset + self.ParametersStr + "/"
        try:
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.makedirs(outputDir)
                print("Create directory:", outputDir)
        except:
            print("Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self, ):
        self.phi_zv = self.K*[[float(0.0)]*self.V]
        for k in range(self.K):
            n_z_sum = 0
            cluster = self.kToClusterPos[k]
            for v in range(self.V):
                n_z_sum += self.n_zv[cluster][v]
            for v in range(self.V):
                self.phi_zv[k][v] = (self.n_zv[cluster][v] + self.beta) / (n_z_sum + self.beta0)

    def getTop(self, array, rankList, Cnt):
        index = 0
        scanned = []
        max = sys.float_info.min
        m = 0
        while m < Cnt and m < len(array):
            max = sys.float_info.min
            for no in range(len(array)):
                if (array[no] >= max and no not in  scanned):
                    index = no
                    max = array[no]
            scanned.append(index)
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        rankList = []
        for k in range(self.K):
            topicline = "Topic " + str(k) + ":"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
            rankList.clear()
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        line = ""
        topicCountIntList = []
        for k in range(self.K):
            cluster = self.kToClusterPos[k]
            topicCountIntList.append([k, self.m_z[cluster]])
        line = ""
        topicCountIntList.sort(key = lambda tc: tc[1], reverse = True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(documentSet.D):
            cluster = self.z[d]
            k = self.clusterPosTok[cluster]
            writer.write(str(k) + "\n")
        writer.close()

    def gibbsSamplingIter(self, documentSet, iterNum, sampleNum, outputPath, wordList):
        parameterList = []
        timeList = []
        niter = 0
        while 1:
            if niter > iterNum:
                break
            if niter == 0:
                ParametersStr = "K" + str(self.Kin) + "iterNum" + str(niter) + "SampleNum" + str(sampleNum) + \
                                "alpha" + str(round(self.alpha, 3)) + "beta" + str(round(self.beta, 3))
                self.output(documentSet, outputPath, wordList)
            if niter > 0 and niter <= 30:
                startTime = time.time()
                self.gibbsOneIteration(documentSet)
                endTime = time.time()
                parameterList.append(niter)
                timeList.append(int(endTime - startTime))
                ParametersStr = "K" + str(self.Kin) + "iterNum" + str(niter) + "SampleNum" + str(sampleNum) + \
                                "alpha" + str(round(self.alpha, 3)) + "beta" + str(round(self.beta, 3))
                self.output(documentSet, outputPath, wordList)
            if niter > 30 and niter <= 200:
                if niter % 5 == 0:
                    startTime = time.time()
                    self.gibbsOneIteration(documentSet)
                    endTime = time.time()
                    parameterList.append(niter)
                    timeList.append(int(endTime - startTime))
                    ParametersStr = "K" + str(self.Kin) + "iterNum" + str(niter) + "SampleNum" + str(sampleNum) + \
                                    "alpha" + str(round(self.alpha, 3)) + "beta" + str(round(self.beta, 3))
                    self.output(documentSet, outputPath, wordList)
                else:
                    self.gibbsOneIteration(documentSet)
            if niter > 200 and niter <= 1000:
                if niter % 100 == 0:
                    startTime = time.time()
                    self.gibbsOneIteration(documentSet)
                    endTime = time.time()
                    parameterList.append(niter)
                    timeList.append(int(endTime - startTime))
                    ParametersStr = "K" + str(self.Kin) + "iterNum" + str(niter) + "SampleNum" + str(sampleNum) + \
                                    "alpha" + str(round(self.alpha, 3)) + "beta" + str(round(self.beta, 3))
                    self.output(documentSet, outputPath, wordList)
                else:
                    self.gibbsOneIteration(documentSet)
            '''
            if niter % 100 == 0:
                print("IterNo:", niter)
            '''
            print("IterNo:", niter)
            niter += 1
        fileParametersStr = "MStreamDiffIter" + "K" + str(self.Kin) + "SampleNum" + str(sampleNum) + "alpha" + \
                            str(round(self.alpha, 3)) + "beta" + str(round(self.beta, 3))
        outTimePath = str(outputPath) + "Time" + str(self.dataset) + str(fileParametersStr) + ".txt"
        writer = open(outTimePath, 'w')
        temp_obj = {}
        temp_obj['parameter'] = parameterList
        temp_obj['Time'] = timeList
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
        writer.close()



