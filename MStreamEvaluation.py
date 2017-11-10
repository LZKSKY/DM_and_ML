# -*- coding: utf-8 -*-
import codecs
import simplejson as json
from sklearn import metrics
import pylab as py
import numpy as np
import math

def MeanAndVar(dataList):
    mean = sum(dataList)*1.0 / len(dataList)
    varience = math.sqrt(sum((mean - value) ** 2 for value in dataList)*1.0 / len(dataList))
    return (mean, varience)

class ClusterEvaluation():
    def __init__(self, resultFilePath):
        self.labelsTrue = []
        self.tweetsCleaned = []
        self.labelsPred = []
        self.resultFilePath = resultFilePath
        self.AMIList = []
        self.NMIList = []
        self.MIList = []
        self.ARIList = []
        self.homogeneityList = []
        self.completenessList = []
        self.VList = []
        self.SCList = []
        
        self.AMITopicKList = []
        self.NMITopicKList = []
        self.MITopicKList = []
        self.ARITopicKList = []
        self.homogeneityTopicKList = []
        self.completenessTopicKList = []
        self.VTopicKList = []
        self.SCTopicKList = []
        self.docNum = 0
              
    def evaluatePerSample(self, sampleNo):
        AMI = metrics.adjusted_mutual_info_score(self.labelsTrue, self.labelsPred)  
        NMI = metrics.normalized_mutual_info_score(self.labelsTrue, self.labelsPred)  
        MI = metrics.mutual_info_score(self.labelsTrue, self.labelsPred)  
        ARI = metrics.adjusted_rand_score(self.labelsTrue, self.labelsPred)  
        homogeneity = metrics.homogeneity_score(self.labelsTrue, self.labelsPred)  
        completeness = metrics.completeness_score(self.labelsTrue, self.labelsPred)  
        V = metrics.v_measure_score(self.labelsTrue, self.labelsPred)    
#        SC = metrics.silhouette_score(self.X, self.labelsPred, metric='sqeuclidean') #Silhouette Coefficient
        self.AMIList.append(AMI)
        self.NMIList.append(NMI)
        self.MIList.append(MI)
        self.ARIList.append(ARI)
        self.homogeneityList.append(homogeneity)
        self.completenessList.append(completeness)
        self.VList.append(V)
        #        self.SCList.append(SC)
        
    def evaluateAllSamples(self, K):
        self.ARITopicKList.append(MeanAndVar(self.ARIList))
        self.MITopicKList.append(MeanAndVar(self.MIList))
        self.AMITopicKList.append(MeanAndVar(self.AMIList))
        self.NMITopicKList.append(MeanAndVar(self.NMIList))
        self.homogeneityTopicKList.append(MeanAndVar(self.homogeneityList))
        self.completenessTopicKList.append(MeanAndVar(self.completenessList))
        self.VTopicKList.append(MeanAndVar(self.VList))
        
        self.AMIList = []
        self.NMIList = []
        self.MIList = []
        self.ARIList = []
        self.homogeneityList = []
        self.completenessList = []
        self.VList = []

    def drawEvaluationResult(self, KRange, Xlabel, titleStr):
        ARIVarianceList = [item[1] for item in self.ARITopicKList]
#        MIVarianceList = [item[1] for item in self.MITopicKList]
        AMIVarianceList = [item[1] for item in self.AMITopicKList]
        NMIVarianceList = [item[1] for item in self.NMITopicKList]
        homogeneityVarianceList = [item[1] for item in self.homogeneityTopicKList]
        completenessVarianceList = [item[1] for item in self.completenessTopicKList]
        VVarianceList = [item[1] for item in self.VTopicKList]
        with open(self.resultFilePath, 'a') as fout:
            fout.write('KRange/iterNumRange:' + repr(KRange) + '\n')
            fout.write('ARIVarianceList:' + repr(ARIVarianceList) + '\n')
            fout.write('AMIVarianceList:' + repr(AMIVarianceList) + '\n')
            fout.write('NMIVarianceList:' + repr(NMIVarianceList) + '\n')
            fout.write('homogeneityVarianceList:' + repr(homogeneityVarianceList) + '\n')
            fout.write('completenessVarianceList:' + repr(completenessVarianceList) + '\n')
            fout.write('VVarianceList:' + repr(VVarianceList) + '\n')
            fout.write('\n')

        ARIMeanList = [item[0] for item in self.ARITopicKList]
#        MIMeanList = [item[0] for item in self.MITopicKList]
        AMIMeanList = [item[0] for item in self.AMITopicKList]
        NMIMeanList = [item[0] for item in self.NMITopicKList]
        homogeneityMeanList = [item[0] for item in self.homogeneityTopicKList]
        completenessMeanList = [item[0] for item in self.completenessTopicKList]
        VMeanList = [item[0] for item in self.VTopicKList]
        with open(self.resultFilePath, 'a') as fout:
            fout.write('KRange/iterNumRange:' + repr(KRange) + '\n')
            fout.write('ARIMeanList:' + repr(ARIMeanList) + '\n')
            fout.write('AMIMeanList:' + repr(AMIMeanList) + '\n')
            fout.write('NMIMeanList:' + repr(NMIMeanList) + '\n')
            fout.write('homogeneityMeanList:' + repr(homogeneityMeanList) + '\n')
            fout.write('completenessMeanList:' + repr(completenessMeanList) + '\n')
            fout.write('VMeanList:' + repr(VMeanList) + '\n')
            fout.write('\n')
            
        with open(self.resultFilePath, 'a') as fout:
            fout.write('\n#%s\n' % (titleStr)) 
            fout.write('\n\n#ARI\n')  
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(ARIMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], ARIMeanList[j], ARIVarianceList[j]))
            fout.write('\n\n#AMI\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(AMIMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], AMIMeanList[j], AMIVarianceList[j]))
            fout.write('\n\n#NMI\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(NMIMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], NMIMeanList[j], NMIVarianceList[j]))
            fout.write('\n\n#homogeneity\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(homogeneityMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], homogeneityMeanList[j], homogeneityVarianceList[j]))
            fout.write('\n\n#completeness\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(completenessMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], completenessMeanList[j], completenessVarianceList[j]))
            fout.write('\n\n#V\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(VMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], VMeanList[j], VVarianceList[j]))

        p1 = py.plot(KRange, ARIMeanList, 'r-*')
        p2 = py.plot(KRange, AMIMeanList, 'g-D')
        p3 = py.plot(KRange, NMIMeanList, 'g-s')
        p4 = py.plot(KRange, homogeneityMeanList, 'b-v')
        p5 = py.plot(KRange, completenessMeanList, 'b-^')
        p6 = py.plot(KRange, VMeanList, 'r-s')
        py.errorbar(KRange, ARIMeanList, yerr=ARIVarianceList, fmt='r*')
        py.errorbar(KRange, AMIMeanList, yerr=AMIVarianceList, fmt='gD')
        py.errorbar(KRange, NMIMeanList, yerr=NMIVarianceList, fmt='gs')
        py.errorbar(KRange, homogeneityMeanList, yerr=homogeneityVarianceList, fmt='bv')
        py.errorbar(KRange, completenessMeanList, yerr=completenessVarianceList, fmt='b^')
        py.errorbar(KRange, VMeanList, yerr=VVarianceList, fmt='rs')
        
        py.legend([p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]], ['ARI', 'AMI', 'NMI', 'Homogeneity','Completeness', 'V'])
        py.xlabel(Xlabel)
        py.ylabel('Performance')      
        py.title(titleStr)
        py.grid(True)
        py.show()
        
#    def getMStreamLabels(self, inFile):
#        self.labelsPred = []
#        self.labelsTrue = []
#        with codecs.open(inFile, 'r') as fin:
#            for lineJson in fin:
#                try:
#                    resultObj = json.loads(lineJson)
#                    self.labelsTrue.append(resultObj['trueCluster'])
#                    self.labelsPred.append(resultObj['predictedCluster'])
#                except:
#                    print inFile
#                    print 'hello world'
#                    print lineJson
                    
    def getMStreamLabels(self, inFile, dataset):
        self.labelsPred = []
        self.labelsTrue = []
        docs = []
        outFile = inFile + "Full.txt"
        with codecs.open(inFile, 'r') as fin:
            for clusterNo in fin:
                try:
                    self.labelsPred.append(int(clusterNo.strip()))
                except:
                    print clusterNo
                    
        with codecs.open(dataset, 'r') as fin:
            for docJson in fin:
                try:
                    docObj = json.loads(docJson)
                    self.labelsTrue.append(docObj['cluster'])
                    docs.append(docObj['text'])
#                    clusterNames.append(docObj['clusterName'])
                except:
                    print docJson
        
        with codecs.open(outFile, 'w') as fout:
            for i in range(len(docs)):
                docObj = {}
                docObj['trueCluster'] = self.labelsTrue[i]
                docObj['predictedCluster'] = self.labelsPred[i]
                docObj['text'] = docs[i]
#                docObj['clusterName'] = clusterNames[i]
                
                docJson = json.dumps(docObj);
                fout.write(docJson + '\n')
                    
    def getMStreamKPredNum(self, inFile):
        labelsPred = []
        with codecs.open(inFile, 'r') as fin:
            for lineJson in fin:
                resultObj = json.loads(lineJson)
                labelsPred.append(resultObj['predictedCluster'])
        KPredNum = np.unique(labelsPred).shape[0]
        return KPredNum
    
    def getPredNumThreshold(self, inFile, Kthreshold):
        KPredNum = 0
        docRemainNum = 0
        docTotalNum = 0
        with codecs.open(inFile, 'r', 'utf-8') as fin:
            clusterSizeStr = fin.readline()
        clusterSizeList = clusterSizeStr.split(',\t')   
#        print clusterSizeList
        for clusterSize in clusterSizeList:
            try:
                clusterSizeCouple = clusterSize.split(':')
                docTotalNum += int(clusterSizeCouple[1])
                if int(clusterSizeCouple[1]) > Kthreshold:
                    KPredNum += 1
                    docRemainNum += int(clusterSizeCouple[1])
            except:
                pass
        return (KPredNum,docRemainNum,docTotalNum)
    
    def docNumPerCluster(self, inFile, parameter, sampleNo, resultFile):
        labelsPred = []
        labelsTrue = []
        with codecs.open(inFile, 'r') as fin:
            for lineJson in fin:
                resultObj = json.loads(lineJson)
                labelsPred.append(resultObj['predictedCluster'])
                labelsTrue.append(resultObj['trueCluster'])
        topicNoVec, indices = np.unique(labelsPred, return_inverse=True)
        # topicNoVec stores the topicNos, sorted by the number of documents in these topics.
        docToTopicVec = topicNoVec[indices]
        # docToTopicVec stores the topicNo for each document.
        
        predTopicNum = len(topicNoVec) 
        maxTopicNo = max(topicNoVec)
        clusterNum = np.unique(self.labelsTrue).shape[0] #Obtain the true number of clusters.
        docNum = len(labelsPred)
        topicSizeVec = [0 for i in range(predTopicNum)]
        for topicNo in indices:
            topicSizeVec[topicNo] += 1

        topicsVec = []
        for topicNo in range(predTopicNum):
            topicTrueNo = topicNoVec[topicNo] #the topicNo in the result file.
            topicsVec.append((topicTrueNo, topicSizeVec[topicNo]))
        topicsVec.sort(key=lambda tup: tup[1], reverse=True)

            
        #find the first and second cluster that each topic relates to.
        topicClusterVec = []
        for i in range(predTopicNum):
            topicNo = topicsVec[i][0] #From topics with most clusters.
            clusterVec = [[clusterNo, 0] for clusterNo in range(clusterNum+1)] 
            #stores the clusters and the number of its documents for each topic.
            for docNo in range(docNum):
                if labelsPred[docNo] == topicNo:
                    clusterNo = int(labelsTrue[docNo])
                    clusterVec[clusterNo][1] += 1
            clusterVec.sort(key=lambda lis: lis[1], reverse=True)
            topicClusterVec.append(clusterVec)
        
        with codecs.open(resultFile, 'a') as fout:
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('predTopicNum=' + str(predTopicNum) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('topicNo:docNum ')
            for tup in topicsVec:
                fout.write(str(tup[0]) + ':' + str(tup[1]) + ', ')
            fout.write('\n')
            
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('predTopicNum=' + str(predTopicNum) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('FirClus:docNum ')     
            for clusterVec in topicClusterVec:
                fout.write('%d:%d, ' % (clusterVec[0][0], clusterVec[0][1]))        
            fout.write('\n')
            
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('predTopicNum=' + str(predTopicNum) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('SecClus:docNum ')     
            for clusterVec in topicClusterVec:
                fout.write('%d:%d, ' % (clusterVec[1][0], clusterVec[1][1]))        
            fout.write('\n')
            
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('predTopicNum=' + str(predTopicNum) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('ThiClus:docNum ')     
            for clusterVec in topicClusterVec:
                fout.write('%d:%d, ' % (clusterVec[2][0], clusterVec[2][1]))        
            fout.write('\n')
            
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('FouClus:docNum ')     
            for clusterVec in topicClusterVec:
                fout.write('%d:%d, ' % (clusterVec[3][0], clusterVec[3][1]))        
            fout.write('\n')
            
def saveTime(Xlabel, XRange, timeList, resultFilePath, titleStr):
    timeMeanList = []
    timeVarianceList = []
    for i in range(len(timeList)):
        timeMeanList.append(np.mean(timeList[i]))
        timeVarianceList.append(np.std(timeList[i]))
    with open(resultFilePath, 'a') as fout:
        fout.write('KRange/iterNumRange:' + repr(XRange) + '\n')
        fout.write('timeList:' + repr(timeList) + '\n')
        fout.write('timeMeanList:' + repr(timeMeanList) + '\n')
        fout.write('timeVarianceList:' + repr(timeVarianceList) + '\n')
        fout.write('\n#%s\n' % (titleStr)) 
        fout.write('\n#IterNum/K\tMean\tSD\n') 
        for j in range(len(timeMeanList)):
            fout.write('%d\t%.4f\t%.4f\n' % (XRange[j], timeMeanList[j], timeVarianceList[j]))
    py.plot(XRange, timeMeanList, 'b-*')
    py.errorbar(XRange, timeMeanList, yerr=timeVarianceList, fmt='b*')
    py.xlabel(Xlabel)
    py.ylabel('Time/sec')      
    py.title(titleStr)
    py.grid(True)
    py.show()

def drawPredK(dataset, resultFilePath, titleStr, Xlabel, XRange, KPredNumMeanList, KPredNumVarianceList):
    with open(resultFilePath, 'a') as fout:
        fout.write('KRange/iterNumRange:' + repr(XRange) + '\n')
        fout.write('KPredNumMeanList:' + repr(KPredNumMeanList) + '\n')
        fout.write('KPredNumVarianceList:' + repr(KPredNumVarianceList) + '\n')
        fout.write('\n#%s\n' % (titleStr)) 
        fout.write('\n#K\tMean\tSD\n') 
        for j in range(len(KPredNumMeanList)):
            fout.write('%.3f\t%.4f\t%.4f\n' % (XRange[j], KPredNumMeanList[j], KPredNumVarianceList[j]))
    
    Ylabel = 'The number of topics fround by MStream'
    py.figure()
    py.plot(XRange, KPredNumMeanList, 'bo')
    py.errorbar(XRange, KPredNumMeanList, yerr=KPredNumVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()

def evaluatePerSample(self, sampleNo):
        AMI = metrics.adjusted_mutual_info_score(self.labelsTrue, self.labelsPred)  
        NMI = metrics.normalized_mutual_info_score(self.labelsTrue, self.labelsPred)  
        MI = metrics.mutual_info_score(self.labelsTrue, self.labelsPred)  
        ARI = metrics.adjusted_rand_score(self.labelsTrue, self.labelsPred)  
        homogeneity = metrics.homogeneity_score(self.labelsTrue, self.labelsPred)  
        completeness = metrics.completeness_score(self.labelsTrue, self.labelsPred)  
        V = metrics.v_measure_score(self.labelsTrue, self.labelsPred)    
#        SC = metrics.silhouette_score(self.X, self.labelsPred, metric='sqeuclidean') #Silhouette Coefficient
        self.AMIList.append(AMI)
        self.NMIList.append(NMI)
        self.MIList.append(MI)
        self.ARIList.append(ARI)
        self.homogeneityList.append(homogeneity)
        self.completenessList.append(completeness)
        self.VList.append(V)
          
def MStreamIterNum():
    K = 0
    iterNumRange = range(1, 31, 1)
    iterNumRangeStr = ''
    sampleNum = 1
    alpha = 1000
    beta = 0.02
    KThreshold = 0
    dataset = 'Tweet'
    datasetPath = 'D:/Research/dataset/Text/' + dataset    
    inPath = 'D:/Project/MStream/result/'
    resultFileName = 'MStreamNoiseKThreshold%dIterNumDataset%sK%dsampleNum%dalpha%.3fbeta%.3fIterNum%s.txt' % (KThreshold, 
        dataset, K, sampleNum, alpha, beta,iterNumRangeStr)
    resultFilePath = 'D:/Project/MStream/result/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)
    
    KPredNumMeanList = []
    KPredNumVarianceList = []    
    noiseNumMeanList = []
    noiseNumVarianceList = []    
    for iterNum in iterNumRange:
        dirName = '%sK%diterNum%dSampleNum%dalpha%.3fbeta%.3f/' % (dataset, K, iterNum, sampleNum, alpha, beta)
        inDir = inPath + dirName
        KPredNumList = []
        noiseNumList = []
        for sampleNo in range(1, sampleNum+1):
            fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
            inFile = inDir + fileName
            MStreamEvaluation.getMStreamLabels(inFile, datasetPath)
            MStreamEvaluation.evaluatePerSample(sampleNo)
            sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
            (KPredNum,docRemainNum,docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
            KPredNumList.append(KPredNum)
            noiseNumList.append(docTotalNum-docRemainNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        noiseNumMeanList.append(np.mean(noiseNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
        noiseNumVarianceList.append(np.std(noiseNumList))
        MStreamEvaluation.evaluateAllSamples(iterNum)
        
#    print 'noiseNumMeanList'
#    print noiseNumMeanList
#    print 'noiseNumVarianceList'
#    print noiseNumVarianceList

    titleStr = 'IterNum to MStream %s K%dSampleNum%dalpha%.3fbeta%.3f' % (dataset, K, sampleNum, alpha, beta)
    Xlabel = 'The number of iterations'
    MStreamEvaluation.drawEvaluationResult(iterNumRange, Xlabel, titleStr)
    
    titlePredK = 'KPredNum%s KThreshold%d K%dSampleNum%dalpha%.3fbeta%.3f' % (dataset,KThreshold, K, sampleNum, alpha, beta)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, iterNumRange, KPredNumMeanList, KPredNumVarianceList)
    
    titleRemainDoc = 'noiseNum%s KThres%d K%dS%dalpha%.3fbeta%.3f' % (dataset,KThreshold, K, sampleNum, alpha, beta)
    drawPredK(dataset, resultFilePath, titleRemainDoc, Xlabel, iterNumRange, noiseNumMeanList, noiseNumVarianceList)
        
    
    timeFilePara = 'Time%sMStreamDiffIterK%dSampleNum%dalpha%.3fbeta%.3f' % (
        dataset, K, sampleNum, alpha, beta)
    timeFilePath = inPath + timeFilePara + '.txt'
    parameterList = []
    timeMeanList = []
    timeVarianceList = []
    timeList = []
    with codecs.open(timeFilePath, 'r', 'utf-8') as fin:
        for timeJson in fin:
            try:    
                timeObj = json.loads(timeJson)
                timeList.append([ timeRun*1.0/1000 for timeRun in timeObj['Times']])
                parameterList = timeObj['parameters']
            except:
                print timeJson
    timeSampleList = [ [0 for j in range(sampleNum)] for i in range(len(parameterList))]
    for j in range(sampleNum):
        for i in range(len(parameterList)):
            timeSampleList[i][j] = timeList[j][i]
    
    for i in range(len(parameterList)):
        (timeMean, timeVariance) = MeanAndVar(timeSampleList[i])
        timeMeanList.append(timeMean)
        timeVarianceList.append(timeVariance)
    
    Xlabel = 'Iterations'            
    Ylabel = 'Time'
    titleStr = 'MStream Running time with different iterations'
    py.figure()
    py.plot(parameterList, timeMeanList, 'b-o')
    py.errorbar(parameterList, timeMeanList, yerr=timeVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()
    
    
def MStreamBeta():
    K = 0
#    betaRange = [i*0.01 for i in range(1, 10, 1)] + [i*0.1 for i in range(1, 11, 1)]
    betaRange = [i*0.01 for i in range(1, 11, 1)]
#    betaRangeStr = '0.01range(1, 11, 1)0.1range(1, 11, 1)'
    betaRangeStr = ''
    sampleNum = 1
    iterNum = 30
    alpha = 0.1
    KThreshold = 5
    dataset = 'Tweet'
    datasetPath = 'D:/Research/dataset/Text/' + dataset
    inPath = 'D:/Project/MStream/result/'
    resultFileName = 'MStreamKThreshold%dBetaDataset%salpha%.3fK%dIterNum%dBetaRange%ssampleNum%d.txt' % (KThreshold, dataset,alpha, K,iterNum, betaRangeStr, sampleNum)
    resultFilePath = 'D:/Project/MStream/result/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []    
    for beta in betaRange:
        dirName = '%sK%diterNum%dSampleNum%dalpha%.3fbeta%.3f/' % (dataset, K, iterNum, sampleNum, alpha, beta)
        inDir = inPath + dirName
        KPredNumList = []
        for sampleNo in range(1, sampleNum+1):
            fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
            inFile = inDir + fileName
            MStreamEvaluation.getMStreamLabels(inFile, datasetPath)
            MStreamEvaluation.evaluatePerSample(sampleNo)
            sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
            (KPredNum,docRemainNum,docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
            KPredNumList.append(KPredNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
        MStreamEvaluation.evaluateAllSamples(iterNum)
    titleStr = 'beta to MStream on %s K%dSampleNum%dalpha%.3fIter%d' % (dataset, K, sampleNum, alpha,  iterNum)
    Xlabel = 'beta'
    MStreamEvaluation.drawEvaluationResult(betaRange, Xlabel, titleStr)
    titlePredK = '%s KThreshold%d K%dSampleNum%dalpha%.3fIter%d' % (dataset, KThreshold, K, sampleNum, alpha,  iterNum)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, betaRange, KPredNumMeanList, KPredNumVarianceList)
    
    timeFilePara = 'Time%sMStreamDiffBetaK%diterNum%dSampleNum%dalpha%.3f' % (dataset, K, iterNum, sampleNum, alpha)
    timeFilePath = inPath + timeFilePara + '.txt'
    parameterList = []
    timeMeanList = []
    timeVarianceList = []
    with codecs.open(timeFilePath, 'r', 'utf-8') as fin:
        for timeJson in fin:
            try:
                timeObj = json.loads(timeJson)
                parameterList.append(timeObj['parameter'])
                (timeMean, timeVariance) = MeanAndVar([ timeRun*1.0/1000 for timeRun in timeObj['Time']])
                timeMeanList.append(timeMean)
                timeVarianceList.append(timeVariance)
            except:
                print timeJson
    Xlabel = 'Beta'            
    Ylabel = 'Time'
    titleStr = 'MStream Running time with different betas'
    py.figure()
    py.plot(parameterList, timeMeanList, 'bo')
    py.errorbar(parameterList, timeMeanList, yerr=timeVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()
    
def MStreamAlpha():
    K = 0
    docNum = 2472
    alphaRange = [i*0.1*docNum for i in range(1, 11, 1)]
    alphaRangeStr = ''
    sampleNum = 1
    iterNum = 30
    beta = 0.02
    KThreshold = 0   
    dataset = 'Tweet'
    datasetPath = 'D:/Research/dataset/Text/' + dataset    
    inPath = 'D:/Project/MStream/result/'
    
    resultFileName = 'MStreamAlphaKThreshold%dDataset%sbeta%.3fK%dIterNum%dAlphaRange%ssampleNum%d.txt' % (KThreshold, dataset, beta, K,iterNum, alphaRangeStr, sampleNum)
    resultFilePath = 'D:/Project/MStream/result/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []    
    for alpha in alphaRange:
        dirName = '%sK%diterNum%dSampleNum%dalpha%.3fbeta%.3f/' % (dataset, K, iterNum, sampleNum, alpha, beta)
        inDir = inPath + dirName
        KPredNumList = []
        for sampleNo in range(1, sampleNum+1):
            fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
            inFile = inDir + fileName
            MStreamEvaluation.getMStreamLabels(inFile, datasetPath)
            MStreamEvaluation.evaluatePerSample(sampleNo)
            sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
            (KPredNum,docRemainNum,docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
            KPredNumList.append(KPredNum)
        MStreamEvaluation.evaluateAllSamples(iterNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
    titleStr = 'alpha to MStream on  %s K%dSampleNum%dbeta%.3fIter%d' % (dataset, K, sampleNum, beta,  iterNum)
    Xlabel = 'alpha'
    titlePredK = '%s KThreshold%d K%dSampleNum%dbeta%.3fIter%d' % (dataset, KThreshold, K, sampleNum, beta,  iterNum)
    MStreamEvaluation.drawEvaluationResult(alphaRange, Xlabel, titleStr)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, alphaRange, KPredNumMeanList, KPredNumVarianceList)
    
    timeFilePara = 'Time%sMStreamDiffAlphaK%diterNum%dSampleNum%dbeta%.3f' % (dataset, K, iterNum, sampleNum, beta)
    timeFilePath = inPath + timeFilePara + '.txt'
    parameterList = []
    timeMeanList = []
    timeVarianceList = []
    with codecs.open(timeFilePath, 'r', 'utf-8') as fin:
        for timeJson in fin:
            try:
                timeObj = json.loads(timeJson)
                parameterList.append(timeObj['parameter'])
                (timeMean, timeVariance) = MeanAndVar([ timeRun*1.0/1000 for timeRun in timeObj['Time']])
                timeMeanList.append(timeMean)
                timeVarianceList.append(timeVariance)
            except:
                print timeJson
    Xlabel = 'Alpha'            
    Ylabel = 'Time'
    titleStr = 'MStream Running time with different alphas'
    py.figure()
    py.plot(parameterList, timeMeanList, 'bo')
    py.errorbar(parameterList, timeMeanList, yerr=timeVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()
    
if __name__ == '__main__':
#    MStreamIterNum()
#    MStreamBeta()
    MStreamAlpha()
