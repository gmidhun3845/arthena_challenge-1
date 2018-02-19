import os
import numpy as np

# convert simple html to json
def convert_to_json(text,index):
    state = 0
    key = ""
    key2 = ""
    data_json = {}
    subresult = ""
    old = index
    while index<len(text):
        l = text[index]
        index+=1
        if state==0:
            if l=='<':
                state=1
        elif state==1:
            if l=="/":
                s = ""
                if not bool(data_json):
                    for i in range(old,index-2):
                        s+=text[i]
                    return s.strip(),index-2
                return data_json,index-2
            elif l=='>':
                subresult,index=convert_to_json(text,index)
                state=2
            else:
                key+=l
        elif state==2:
            if l=='<':
                state=3
        else:
            if l=='>':
                if key in data_json:
                    c = 1
                    while key+"_"+str(c)  in data_json:
                        c+=1
                    key = key+"_"+str(c)
                data_json[key] = subresult
                key = ""
                key2= ""
                subresult = ""
                state = 0
            else:
                key2+=l
    return data_json,index

# read file and pass the data to convert json
def extractInfo(f):
    text = open(f).read()
    data_json,_ = convert_to_json(np.array(list(text.strip())),0)
    return data_json

# Data for Task
def getfilenames(path):
    filepaths = []
    for paths,folders,files in os.walk(path):
        if len(files)>0:
            for file in files:
                if file.endswith('.html'):
                    filepaths.append(paths+'/'+file)
    return filepaths

def script1(filepaths,whereto):
    result = []
    for file in filepaths:
        data = extractInfo(file)
        result.append(access(data,whereto['artist']))
    return result

def access(data,wheretolist):
    temp = data
    for l in wheretolist:
        temp = temp[l]
    return temp

def prepData(file,artist_index,artist_count,result,whereto):
    data = extractInfo(file)
    temp = {}
    temp['artist'] = access(data,whereto['artist'])
    temp['works'] = []
    index = artist_count
    if temp['artist'] not in artist_index:
        result.append(temp)
        artist_index[temp['artist']] = artist_count
        artist_count+=1
    else:
        index = artist_index[temp['artist']]
    return artist_index,artist_count,index,data


def script2(filepaths,whereto):
    result = []
    artist_index = {}
    artist_count = 0
    for file in filepaths:
        artist_index,artist_count,index,data = prepData(file,artist_index,artist_count,result,whereto)
        result[index]['works'].append(access(data,whereto['title']))
    return result

def script3(filepaths,whereto):
    result = []
    artist_index = {}
    artist_count = 0
    for file in filepaths:
        artist_index,artist_count,index,data = prepData(file,artist_index,artist_count,result,whereto)
        result[index]['works'].append({'title': access(data,whereto['title']), 'price': access(data,whereto['price'])})
    return result

def script4(filepaths,whereto):
    result = []
    artist_index = {}
    artist_count = 0
    for file in filepaths:
        artist_index,artist_count,index,data = prepData(file,artist_index,artist_count,result,whereto)
        currency, amount = access(data,whereto['price']).split()
        result[index]['works'].append({'title': access(data,whereto['title']), 'currency': currency, 'amount': amount })
    return result

def script5(filepaths,whereto):
    result = []
    artist_index = {}
    artist_count = 0
    for file in filepaths:
        artist_index,artist_count,index,data = prepData(file,artist_index,artist_count,result,whereto)
        result[index]['works'].append({'title': access(data,whereto['title']), 'currency': access(data,whereto['currency']), 'amount': access(data,whereto['amount']) })
    return result

filepaths = getfilenames("lot-parser/data/2015-03-18/")

print("Reading the directory 2015-03-18 : ","\n")

whereto = {"artist": ["html","body","h2"]}
result1 = script1(filepaths,whereto)
print("Results of Task1 : ",result1,"\n")

whereto = {"artist": ["html","body","h2"],"title":["html","title"]}
result2 = script2(filepaths,whereto)
print("Results of Task2 : ",result2,"\n")

whereto = {"artist": ["html","body","h2"],"title":["html","title"],"price":["html","body","div_1"]}
result3 = script3(filepaths,whereto)
print("Results of Task3 : ",result3,"\n")

whereto = {"artist": ["html","body","h2"],"title":["html","title"],"price":["html","body","div_1"]}
result4 = script4(filepaths,whereto)
print("Results of Task4 : ",result4,"\n")

filepaths = getfilenames("lot-parser/data/2017-12-20/")

print("Reading the directory 2017-12-20 : ","\n")

whereto = {"artist": ["html","body","h3 class=\'artist\'"],"title":["html","title"],"currency":["html","body","div_1","span class=\'currency\'"],"amount":["html","body","div_1","span"]}
result5 = script5(filepaths,whereto)
print("Results of Task4 : ",result5,"\n")

final_data = []
artist_count = 0
artist_index = {}
for l in result4:
    l['artist'].replace("(.*)","")
    l['artist'].strip()
    if l['artist'] in artist_index:
        for work in l['works']:
            final_data[artist_index[l['artist']]]['works'].append(work)
    else:
        final_data.append(l)
        artist_index[l['artist']] = artist_count
        artist_count+=1

for l in final_data:
    sum = 0
    for work in l['works']:
        if work['currency']=='USD':
            sum += float(work['amount'].replace(',',''))
        if work['currency']=='GBP':
            sum += float(work['amount'].replace(',',''))*1.34
    l['totalValue'] = "USD "+str(int(sum))
print("Final Results : ",final_data,"\n")