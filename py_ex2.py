import operator
import math
import re
from functools import reduce
import copy


originalDic = {}

def LoadData(fileTrain, fileTest):
    ''' Load dataset
       return trainset, testset and attributes '''

    global g_attributes
    train_set = list()
    test_set = list()
    with open(str(fileTrain), 'r') as ftrain_handle:
        line = ftrain_handle.readline()
        g_attributes = re.split('\t|\n',line)[:-1]
        train_set = ftrain_handle.readlines()
        train_set = list(re.split('\t|\n',x)[:-1] for x in train_set)

    with open(str(fileTest), 'r') as ftest_handle:
        ftest_handle.readline()
        test_set = ftest_handle.readlines()
        test_set = list(re.split('\t|\n',x)[:-1] for x in test_set)

    return train_set, test_set, g_attributes

def getAccuracy(testSet, predictions):
    ''' Get accuracy of predictions in relation to accept classification '''
    lenghtTest = len(testSet)
    correctClasfiy = 0
    for x in range(0,lenghtTest):
        if testSet[x][-1] == predictions[x]:
            correctClasfiy += 1
    return (correctClasfiy/float(lenghtTest))*100.0


def Hamming(observe1, observe2):
    ''' Calculate hamming destination between two oberves '''
    count = 0
    for x in observe1:
        if x not in observe2:
            count += 1
    return count

def getNeighbors(test_data, trainSet, k, length):
    ''' Get K neighbors of given data '''
    distances = []
    for y in range (1,length):
        dist = Hamming(test_data, trainSet[y][:-1])
        distances.append((trainSet[y], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
         neighbors.append(distances[i][0])
    return neighbors

def getClassification(neighbors):
    ''' Vote for the most frequent classification '''
    votes = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1

    class_vote = max(votes.items(), key=operator.itemgetter(1))[0]
    return class_vote


def KNN(trainSet, testSet, k):
    ''' KNN algorithm '''
    predictions = list()
    lengthTrain = len(trainSet)
    lengthTest = len(testSet)

    for x in range(0,lengthTest):
        neighbors = getNeighbors(testSet[x][:-1], trainSet, k, lengthTrain)
        class_vote = getClassification(neighbors)
        predictions.append(class_vote)
    
    return predictions




def BuildProbs(trainSet, testSet, len_test_set, lableList, attributeList):
    ''' Build probabilities for NB algorithm '''
    predictions = []
    for j in range(0,len_test_set):
        observe = testSet[j]
        probabilities = {}
        for lable in lableList:
            attributes = []
            for i in range(0,len(observe)-1):
                s = sum(1 for x in lableList[lable] if x[i] == observe[i]) + 1
                k = len(attributeList[i])
                attributes.append([s,k])
            probs = list( (i[0]/float( len(lableList[lable]) + i[1] )) for i in attributes)
            probabilities[lable] = (reduce(lambda x, y: x*y, probs)) * (len(lableList[lable])/float(len(trainSet)))
        class_prob = sorted(probabilities.items(), key=operator.itemgetter(1), reverse = True)
        predictions.append(class_prob[0][0])
    return predictions



def PrepareDataForNaiveBaise(trainSet, testSet):
    ''' Build data for NB algorithm '''
    lableList = {}
    attributeList = list()
    len_train_set = len(trainSet)
    len_test_set = len(testSet)

    for j in range(0,len(trainSet[0])):
        attributeList.append(set())
    
    for i in range(0, len_train_set):
        example = trainSet[i]
        for j in range(0,len(example)):
            attributeList[j].add(example[j])
        if (example[-1] not in lableList):
            lableList[example[-1]] = []
        lableList[example[-1]].append(example)
    
    return BuildProbs(trainSet, testSet, len_test_set, lableList, attributeList)



def NaiveBaise(trainSet, testSet):
    ''' NaiveBaise algorithm '''
    return PrepareDataForNaiveBaise(trainSet, testSet)






def calc_entropy(data):
    ''' Calculate entropy of data '''
    entries = len(data) 
    labels = {}
    for rec in data:
        label = rec[-1] 
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    entropy = 0.0
    for key in labels:
        prob = float(labels[key])/entries
        entropy -= prob * math.log(prob, 2) 
    return entropy

def attribute_selection(data):
    ''' Choose the best attribute '''
    features = len(data[0]) - 1
    baseEntropy = calc_entropy(data)
    max_InfoGain = 0.0
    bestAttr = 0
    for i in range(features):
        AttrList = [rec[i] for rec in data]
        uniqueVals = set(AttrList)
        newEntropy = 0.0
        attrEntropy = 0.0 
        for value in uniqueVals:
            newData = dataset_split(data, i, value) 
            prob = len(newData)/float(len(data)) 
            newEntropy = prob * calc_entropy(newData) 
            attrEntropy += newEntropy
        infoGain = baseEntropy - attrEntropy
        if (infoGain >= max_InfoGain):
            max_InfoGain = infoGain
            bestAttr = i
    return bestAttr

def Mode(data):
    ''' Choose the major class of data '''
    arr = {}

    for row in data:
        x = row[-1]
        if x not in arr:
            arr[x] = 1
        arr[x] += 1

    mode = max(arr.items(), key=lambda item: item[1])[0]

    return mode

def dataset_split(data, arc, val):
    ''' Split the data by the best attribute '''
    newData = []
    for rec in data: 
        if rec[arc] == val:
            reducedSet = list(rec[:arc]) 
            reducedSet.extend(rec[arc+1:])
            newData.append(reducedSet)
    return newData



def indexOfBestAttrOrgTrainSet(attribute_name):
    ''' Return the index of the best attribute of the original trainingset '''
    return originalDic[attribute_name]


def decision_tree(trainSet, attributes, default, originalTrainSet):
    ''' Return the dicision tree of trainingset '''
    if not trainSet:
        return default
    labels = [row[-1] for row in trainSet]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(attributes) == 1:
        return Mode(trainSet)
    else:
        bestAttrubute = attribute_selection(trainSet)
        attribute_name = attributes[bestAttrubute]
        tree = {attribute_name:{}}
        del(attributes[bestAttrubute])
        nodes = [row[bestAttrubute] for row in trainSet]
        uniq_values = set(nodes)
        for value in uniq_values:
            subAttributes = copy.deepcopy(attributes)
            newData = dataset_split(trainSet, bestAttrubute, value)
            sub_tree = decision_tree(newData, subAttributes, Mode(trainSet), originalTrainSet)
            tree[attribute_name][value] = sub_tree
        index = indexOfBestAttrOrgTrainSet(attribute_name)
        nodes_g = [row[index] for row in originalTrainSet]
        uniq_values_g = set(nodes_g)
        for attr in uniq_values_g:
            if attr not in uniq_values:
                sub_tree = decision_tree({}, attributes, Mode(trainSet), originalTrainSet)
                tree[attribute_name][attr] = sub_tree
    return tree


def DTL(trainSet, attributes, testSet):
    ''' ID3 algorithm '''
    tree = decision_tree(trainSet, copy.deepcopy(attributes) , Mode(trainSet), trainSet)  

    strr = printTree(tree)
    with open("output_tree.txt", 'w') as fhandle:
        fhandle.write(strr)
    predictions = list()
    for e in testSet:
        predictions.append(predictTree(tree, copy.deepcopy(attributes), e[:-1]))
    return predictions

def predictTree(tree, attributes, observe):
    ''' Predict test example by given dicision tree '''
    if type(tree) is str:
        return tree
    attr = list(tree.keys())[0]
    val = observe[attributes.index(attr)]
    return predictTree(tree[str(attr)][str(val)], attributes, observe)

def printTree(tree, depth = 0):
    ''' Print given dicision tree '''
    lines=""
    attribute = list(tree.keys())[0]
    sub_tree =list(tree.values())[0]
    for i, val in enumerate(sorted(sub_tree.keys())):
        if i!=0:
            lines += '\t' * (depth -1)
        tree = sub_tree.get(val)
        if type(tree) is str:
            lines += "|" + attribute + "=" + val + ":" + tree + '\n'
        else:
            if depth == 0:
                lines +=  attribute + "=" + val + '\n'
            else:
                lines += "|" + attribute + "=" + val + '\n'
            lines += '\t' * depth  + printTree(tree, depth + 1)
    return lines

def main():
    fileTrain = "train.txt"
    fileTest = "test.txt"
    train, test, attributes = LoadData(fileTrain, fileTest)
    i = 0
    for attr in attributes:
        originalDic[attr] = i
        i += 1
    predictions = []
    predictions.append(DTL(train, attributes, test))
    predictions.append(KNN(train, test, 5))
    predictions.append(NaiveBaise(train, test))
    with open("output.txt", 'w') as fhandle:
        fhandle.write("{}\t{}\t{}\t{}\n".format("Num", "DT", "KNN", "naiveBase"))
        for i in range(0,len(test)):
            fhandle.write("{}\t{}\t{}\t{}\n".format(i+1, predictions[0][i], predictions[1][i], predictions[2][i]))
        fhandle.write("\t{}\t{}\t{}".format( math.ceil(getAccuracy(test,predictions[0]))/100, math.ceil(getAccuracy(test,predictions[1]))/100, 
                                                    math.ceil(getAccuracy(test,predictions[2]))/100 ) )
if __name__ == "__main__":
    main()
