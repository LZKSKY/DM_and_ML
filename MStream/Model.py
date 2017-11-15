import random
import os
import time
import json
import copy

class Model:
    KIncrement = 100
    smallDouble = 1e-150
    largeDouble = 1e150
    Max_Batch = 5 # Max number of batches we will consider

    def __init__(self, K, KIncrement, V, iterNum,alpha, beta, dataset, ParametersStr, sampleNo, wordsInTopicNum, timefil):
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

        self.batchNum2tweetID = {} # batch num to tweet id
        self.batchNum = 1 # current batch number
        with open(timefil) as timef:
            for line in timef:
                buff = line.strip().split(' ')
                if buff == ['']:
                    break
                self.batchNum2tweetID[self.batchNum] = int(buff[1])
                self.batchNum += 1
        self.batchNum = 1
        print("There are", self.batchNum2tweetID.__len__(), "time points.\n\t", self.batchNum2tweetID)

    def run(self, documentSet, outputPath, wordList):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = self.K # the number of cluster containing documents currently
        self.BatchSet = {} # Store information of each batch
        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
            if self.batchNum <= self.Max_Batch:
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet)
                self.gibbsSampling(documentSet)
            else:
                # remove influence of batch earlier than Max_Batch
                self.D -= self.BatchSet[self.batchNum - self.Max_Batch]['D']
                for cluster in self.m_z:
                    if cluster in self.BatchSet[self.batchNum - self.Max_Batch]['m_z']:
                        self.m_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['m_z'][cluster]
                        self.checkEmpty(cluster)
                        self.n_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_z'][cluster]
                        for word in self.n_zv[cluster]:
                            if word in self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster]:
                                self.n_zv[cluster][word] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster][word]
                self.BatchSet.pop(self.batchNum - self.Max_Batch)
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet)
                self.gibbsSampling(documentSet)
            # get influence of only the current batch (remove other influence)
            self.BatchSet[self.batchNum-1]['D'] = self.D - self.BatchSet[self.batchNum-1]['D']
            for cluster in self.m_z:
                if cluster not in self.BatchSet[self.batchNum - 1]['m_z']:
                    self.BatchSet[self.batchNum - 1]['m_z'][cluster] = 0
                if cluster not in self.BatchSet[self.batchNum - 1]['n_z']:
                    self.BatchSet[self.batchNum - 1]['n_z'][cluster] = 0
                self.BatchSet[self.batchNum - 1]['m_z'][cluster] = self.m_z[cluster] - self.BatchSet[self.batchNum - 1]['m_z'][cluster]
                self.BatchSet[self.batchNum - 1]['n_z'][cluster] = self.n_z[cluster] - self.BatchSet[self.batchNum - 1]['n_z'][cluster]
                if cluster not in self.BatchSet[self.batchNum - 1]['n_zv']:
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster] = {}
                for word in self.n_zv[cluster]:
                    if word not in self.BatchSet[self.batchNum - 1]['n_zv'][cluster]:
                        self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = 0
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = self.n_zv[cluster][word] - self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word]
            print("\tGibbs sampling successful! Start to saving results.")
            self.output(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")

    def intialize(self, documentSet):
        for d in range(self.currentDoc, self.D_All):
            documentID = documentSet.documents[d].documentID
            if documentID <= self.batchNum2tweetID[self.batchNum]:
                self.D += 1
            else:
                break
        print("\t" + str(self.D) + " documents will be analyze")
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID
            if documentID <= self.batchNum2tweetID[self.batchNum]:
                cluster = self.sampleCluster(d, document)  # Get initial cluster of each document
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_z[cluster] += wordFre
                if d == self.D_All - 1:
                    self.startDoc = self.currentDoc
                    self.currentDoc = self.D_All
                    self.batchNum += 1
            else:
                self.startDoc = self.currentDoc
                self.currentDoc = d
                self.batchNum += 1
                break

    def gibbsSampling(self, documentSet):
        for i in range(self.iterNum):
            print("\titer is ", i)
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID
                cluster = self.z[documentID]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)
                cluster = self.sampleCluster(d, document)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre

    def sampleCluster(self, d, document):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.m_z[cluster] = 0
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster] / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if cluster not in self.n_zv:
                        self.n_zv = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha / (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2
        # 为什么这个样子？
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
            self.K_current += 1
        return kChoosed

    def checkEmpty(self, cluster):
        if self.m_z[cluster] == 0:
            self.K_current -= 1

    def output(self, documentSet, outputPath, wordList, batchNum):
        outputDir = outputPath + self.dataset + self.ParametersStr + "Batch" + str(batchNum) + "/"
        try:
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):
        self.phi_zv = {}
        for cluster in self.n_zv:
            n_z_sum = 0
            if self.m_z[cluster] != 0:
                if cluster not in self.phi_zv:
                    self.phi_zv[cluster] = {}
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        n_z_sum += self.n_zv[cluster][v]
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.beta) / float(n_z_sum + self.beta0)

    def getTop(self, array, rankList, Cnt):
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in array:
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if k not in self.phi_zv:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key = lambda tc: tc[1], reverse = True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(self.startDoc, self.currentDoc):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[documentID]
            writer.write(str(documentID) + " " + str(cluster) + "\n")
        writer.close()

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