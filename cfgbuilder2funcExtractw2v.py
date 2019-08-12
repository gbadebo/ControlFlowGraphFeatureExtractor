from utils import buildCFG
from time import sleep
import numpy as np
from collections import defaultdict

import pandas as pd
import sys
import os.path
import gensim
from collections import Counter
import nltk
import string
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem.porter import PorterStemmer
from time import sleep
import re
#sudo pip install -U Scikit-learn==0.20

featuretype = 'w2v'
def checkFuncInFile(filename):
	content = ""
	try:	
		
		with open(filename, 'r') as content_file:
			content = content_file.read()
		myfunc =["cin","getenv","getenv","wgetenv","wgetenvs","catgets","gets","getchar","getc","getch","getche","kbhit","stdin","getdlgtext","getpass","scanf","fscanf","vscanf","vfscanf","istream.get","istream.getline","istream.peek","istream.read*","istream.putback","streambuf.sbumpc","streambuf.sgetc","streambuf.sgetn","streambuf.snextc","streambuf.sputbackc","SendMessage","SendMessageCallback","SendNotifyMessage","PostMessage","PostThreadMessage","recv","recvfrom","Receive","ReceiveFrom","ReceiveFromEx","Socket.Receive*","memcpy","wmemcpy","memccpy","memmove","wmemmove","memset","wmemset","memcmp","wmemcmp","memchr","wmemchr","strncpy","strncpy*","lstrcpyn","tcsncpy*","mbsnbcpy*","wcsncpy*","wcsncpy","strncat","strncat*","mbsncat*","wcsncat*","bcopy","strcpy","lstrcpy","wcscpy","tcscpy","mbscpy","CopyMemory","strcat","lstrcat","lstrlen","strchr","strcmp","strcoll","strcspn","strerror","strlen","strpbrk","strrchr","strspn","strstr","strtok","strxfrm","readlink","fgets","sscanf","swscanf","sscanfs","swscanfs","printf","vprintf","swprintf","vsprintf","asprintf","vasprintf","fprintf","sprint","snprintf","snprintf*","snwprintf*","vsnprintf","CString.Format","CString.FormatV","CString.FormatMessage","CStringT.Format","CStringT.FormatV","CStringT.FormatMessage","CStringT.FormatMessageV","syslog","malloc","Winmain","GetRawInput*","GetComboBoxInfo","GetWindowText","GetKeyNameText","Dde*","GetFileMUI*","GetLocaleInfo*","GetString*","GetCursor*","GetScroll*","GetDlgItem*","GetMenuItem*","free","delete","new","malloc","realloc","calloc","alloca","strdup","asprintf","vsprintf","vasprintf","sprintf","snprintf","snprintf","snwprintf","vsnprintf"]
		for fun in myfunc:
			if fun.strip() in content:
				return True
		return False
	except:
		"Could not read filename"
		return False	
def writetofile(filename, content):
	with open(filename, "a") as myfile:
		myfile.write(content)
def dictoMatrix(g):
	keys=sorted(g.keys())
	size=len(keys)
	
	M = [ [0]*size for i in range(size) ]
	
	for a,b in [(keys.index(a), keys.index(b)) for a, row in g.items() for b in row]:
		
		M[a][b] = 0 if (a==b) else 1
		
	return M


sourcefiles = "/home/gbaduz/Downloads/pyc-cfg-master/VulDeePecker-master/CWE-119/source_files/"
database = "/home/gbaduz/Downloads/pyc-cfg-master/VulDeePecker-master/CWE-119/CGD/cwe119_cgd.txt"
dataset = []
def extractFilenameLabel(filename):
	with open(filename, 'r') as content_file:
    		content = content_file.read()
		
		data = content.split("---------------------------------")
		for d in data[:-1]:
			label = d[-2]
			filen = d.split(" ")[1]
			if checkFuncInFile(sourcefiles+filen):
				print len(dataset)
				dataset.append([sourcefiles+filen,label])
			
		print len(data) 	
			

extractFilenameLabel(database)
print "Total " + str(len(dataset))
succ = []
failed = []

traindata = []
trainlabel = []
for f in dataset:
	try:
		
		cfgs = buildCFG(f[0])
		print f
		succ.append(f)
		with open(f[0], 'r') as content_file:
			content = content_file.read()
		for cfg in cfgs:
			
			start = content.index(cfg[0]) #pick function name
			start = content.index("{", start)
			stack = []
			funcContent = ""
			for i in range(start,len(content)):
				if content[i] == "{":
					stack.append("{")
				elif content[i] == "}":
					stack.remove(stack[0])
				if len(stack) is 0:
					funcContent = content[start+len(cfg[0]):i]
					break
					     	
			myfunc =["cin","getenv","getenv","wgetenv","wgetenvs","catgets","gets","getchar","getc","getch","getche","kbhit","stdin","getdlgtext","getpass","scanf","fscanf","vscanf","vfscanf","istream.get","istream.getline","istream.peek","istream.read*","istream.putback","streambuf.sbumpc","streambuf.sgetc","streambuf.sgetn","streambuf.snextc","streambuf.sputbackc","SendMessage","SendMessageCallback","SendNotifyMessage","PostMessage","PostThreadMessage","recv","recvfrom","Receive","ReceiveFrom","ReceiveFromEx","Socket.Receive*","memcpy","wmemcpy","memccpy","memmove","wmemmove","memset","wmemset","memcmp","wmemcmp","memchr","wmemchr","strncpy","strncpy*","lstrcpyn","tcsncpy*","mbsnbcpy*","wcsncpy*","wcsncpy","strncat","strncat*","mbsncat*","wcsncat*","bcopy","strcpy","lstrcpy","wcscpy","tcscpy","mbscpy","CopyMemory","strcat","lstrcat","lstrlen","strchr","strcmp","strcoll","strcspn","strerror","strlen","strpbrk","strrchr","strspn","strstr","strtok","strxfrm","readlink","fgets","sscanf","swscanf","sscanfs","swscanfs","printf","vprintf","swprintf","vsprintf","asprintf","vasprintf","fprintf","sprint","snprintf","snprintf*","snwprintf*","vsnprintf","CString.Format","CString.FormatV","CString.FormatMessage","CStringT.Format","CStringT.FormatV","CStringT.FormatMessage","CStringT.FormatMessageV","syslog","malloc","Winmain","GetRawInput*","GetComboBoxInfo","GetWindowText","GetKeyNameText","Dde*","GetFileMUI*","GetLocaleInfo*","GetString*","GetCursor*","GetScroll*","GetDlgItem*","GetMenuItem*","free","delete","new","malloc","realloc","calloc","alloca","strdup","asprintf","vsprintf","vasprintf","sprintf","snprintf","snprintf","snwprintf","vsnprintf"]
			for fun in myfunc:
				if fun.strip() in funcContent:
					traindata.append(funcContent)
					if int(f[1]) == 1: 
						trainlabel.append(str(1.0))
					else:
						trainlabel.append(str(0.0))
					break

	except:
		#print f[0] + "failed"
		failed.append(f)

print len(traindata)
print len(trainlabel)


sourcefiles = "/home/gbaduz/Downloads/pyc-cfg-master/VulDeePecker-master/CWE-399/source_files/"
database = "/home/gbaduz/Downloads/pyc-cfg-master/VulDeePecker-master/CWE-399/CGD/cwe399_cgd.txt"
dataset = []

extractFilenameLabel(database)
print "Total " + str(len(dataset))
succ = []
failed = []


for f in dataset:
	try:
		
		cfgs = buildCFG(f[0])
		print f
		succ.append(f)
		with open(f[0], 'r') as content_file:
			content = content_file.read()
		for cfg in cfgs:
			
			start = content.index(cfg[0]) #pick function name
			start = content.index("{", start)
			stack = []
			funcContent = ""
			for i in range(start,len(content)):
				if content[i] == "{":
					stack.append("{")
				elif content[i] == "}":
					stack.remove(stack[0])
				if len(stack) is 0:
					funcContent = content[start+len(cfg[0]):i]
					break
					     	
			myfunc =["cin","getenv","getenv","wgetenv","wgetenvs","catgets","gets","getchar","getc","getch","getche","kbhit","stdin","getdlgtext","getpass","scanf","fscanf","vscanf","vfscanf","istream.get","istream.getline","istream.peek","istream.read*","istream.putback","streambuf.sbumpc","streambuf.sgetc","streambuf.sgetn","streambuf.snextc","streambuf.sputbackc","SendMessage","SendMessageCallback","SendNotifyMessage","PostMessage","PostThreadMessage","recv","recvfrom","Receive","ReceiveFrom","ReceiveFromEx","Socket.Receive*","memcpy","wmemcpy","memccpy","memmove","wmemmove","memset","wmemset","memcmp","wmemcmp","memchr","wmemchr","strncpy","strncpy*","lstrcpyn","tcsncpy*","mbsnbcpy*","wcsncpy*","wcsncpy","strncat","strncat*","mbsncat*","wcsncat*","bcopy","strcpy","lstrcpy","wcscpy","tcscpy","mbscpy","CopyMemory","strcat","lstrcat","lstrlen","strchr","strcmp","strcoll","strcspn","strerror","strlen","strpbrk","strrchr","strspn","strstr","strtok","strxfrm","readlink","fgets","sscanf","swscanf","sscanfs","swscanfs","printf","vprintf","swprintf","vsprintf","asprintf","vasprintf","fprintf","sprint","snprintf","snprintf*","snwprintf*","vsnprintf","CString.Format","CString.FormatV","CString.FormatMessage","CStringT.Format","CStringT.FormatV","CStringT.FormatMessage","CStringT.FormatMessageV","syslog","malloc","Winmain","GetRawInput*","GetComboBoxInfo","GetWindowText","GetKeyNameText","Dde*","GetFileMUI*","GetLocaleInfo*","GetString*","GetCursor*","GetScroll*","GetDlgItem*","GetMenuItem*","free","delete","new","malloc","realloc","calloc","alloca","strdup","asprintf","vsprintf","vasprintf","sprintf","snprintf","snprintf","snwprintf","vsnprintf"]
			for fun in myfunc:
				if fun.strip() in funcContent:
					traindata.append(funcContent)
					if int(f[1]) == 1: 
						trainlabel.append(str(2.0))
					else:
						trainlabel.append(str(0.0))
					break

	except:
		#print f[0] + "failed"
		failed.append(f)

print len(traindata)
print len(trainlabel)



print Counter(trainlabel)



if featuretype == 'w2v':
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))




#nltk.download('punkt')
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    #text = unicode(text, errors='replace')
    tokens = re.split(r'[;,\s]\s*',text)
    stems = stem_tokens(tokens, stemmer)
    return stems


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english',decode_error='ignore',max_features = 1000)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_) # if word not seen uses this as default that is what lambda is for. idf is global for each word in the corpus. idf means inverse doc freq which is inverse of num of docs that contain the term log N_totalno of doc/freqterm in doc
        self.word2weight = defaultdict(
            lambda: max_idf,#lamba means use this as default if w is not in corpus
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] #* self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])







if featuretype == 'now2v':
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english',decode_error='ignore',max_features = 200)
    
else:
    tfidf = TfidfEmbeddingVectorizer(w2v)
#newdict = data_dict.copy()
#newdict.update(test_data_dict)
tfidf.fit( traindata,trainlabel)
tfs = tfidf.transform(traindata)

if featuretype == 'now2v':
    tfs = tfs.toarray()

mytraindf = pd.DataFrame(tfs)
mytrainlbldf = pd.DataFrame(trainlabel)
datadf = pd.concat([mytraindf,mytrainlbldf],axis=1)
if featuretype == 'now2v':
	datadf.to_csv("alldata.csv",header=False, index=False)
else:
	datadf.to_csv("alldataw2v.csv",header=False, index=False)





