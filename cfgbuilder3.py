from utils import buildCFG
from time import sleep
import numpy as np

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

sourcefiles = "/home/gbaduz/Downloads/pyc-cfg-master/VulDeePecker-master/CWE-399/source_files/"
database = "/home/gbaduz/Downloads/pyc-cfg-master/VulDeePecker-master/CWE-399/CGD/cwe399_cgd.txt"
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
for f in dataset:
	try:
		
		cfgs = buildCFG(f[0])
		print f
		succ.append(f)
		for cfg in cfgs:
		   #content = "\n[+] Function: " +  str(cfg[0])
		   content = cfg[1].printer()
		   dataMatrix = dictoMatrix(content)
		   temp = np.array(dataMatrix).flatten().tolist()
		   data =   " ".join(str(x) for x in temp)
		   
		   writetofile("CWE399/dataset", data + " " + str(f[1]) + "\n")
	except:
		#print f[0] + "failed"
		failed.append(f)

print "success " + str(len(succ))


print "failed " + str(len(failed))

