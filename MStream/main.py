from MStream import MStream
import json
import time

dataDir = "data/"
outputPath = "result/"
dataset = "newtweets"
timefil = "timefil"
docNum = 1000
alpha = 0.01 * docNum
K = 0 # Number of clusters
KIncrement = 100
beta = 0.02
iterNum = 2
sampleNum = 1
wordsInTopicNum = 15

def runMStreamSimple_batch(K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    mstream = MStream(K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    mstream.getDocuments()
    for sampleNo in range(sampleNum):
        print("SampleNo:"+str(sampleNo))
        mstream.runMStream(sampleNo)

def runWithAlphaScale(beta, K, KIncrement, iterNum, sampleNum, dataset, timefil, wordsInTopicNum, docNum):
    parameters = []
    timeArrayOfParas = []
    p = 0.01
    while p <= 0.101:
        alpha = docNum * p
        parameters.append(p)
        print("alpha:", alpha, "\tp:", p)
        mstream = MStream(K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum+1):
            print("SampleNo:", sampleNo)
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        p += 0.01
    p = 0.2
    while p <= 1.01:
        alpha = docNum * p
        parameters.append(p)
        print("alpha:", alpha, "\tp:", p)
        mstream = MStream(K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo)
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        p += 0.1
    fileParameters = "MStreamDiffAlpha" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + \
                     str(sampleNum) + "beta" + str(round(beta, 3))
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i].__len__()
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithBetas(alpha, K, KIncrement, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    beta = 0.01
    while beta <= 0.101:
        parameters.append(beta)
        print("beta:", beta)
        mstream = MStream(K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        beta += 0.01
    beta = 0.2
    while beta <= 1.01:
        parameters.append(beta)
        print("beta:", beta)
        mstream = MStream(K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        beta += 0.1
    fileParameters = "MStreamDiffBeta" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3))
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithNiters(K, KIncrement, alpha, beta, iterNum, sampleNum, eventName, wordsInTopicNum):
    mstream = MStream(K, KIncrement, alpha, beta, iterNum, sampleNum, eventName, wordsInTopicNum)
    mstream.getDocuments()
    sampleNo = 1
    while sampleNo<=sampleNum:
        print("SampleNo:", sampleNo)
        mstream.runMStreamIter(sampleNo)
        sampleNo += 1

if __name__ == '__main__':
    runMStreamSimple_batch(K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithAlphaScale(beta, K, KIncrement, iterNum, sampleNum, dataset, timefil, wordsInTopicNum, docNum)
    # runWithBetas(alpha, K, KIncrement, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithNiters(K, KIncrement, alpha, beta, iterNum, sampleNum, dataset, wordsInTopicNum)