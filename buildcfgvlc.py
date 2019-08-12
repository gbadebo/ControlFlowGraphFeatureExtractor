from utils import buildCFG
from time import sleep

PATH = '/home/gbaduz/Downloads/bash-4.3/'
import os
from glob import glob
result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.c'))]

succ = []
failed = []

for f in result:
	try:
		
		cfgs = buildCFG(f)
		print f
		succ.append(f)
		for cfg in cfgs:
		    print "\n[+] Function: ", cfg[0]
		    print cfg[1].printer()
		sleep(2)
	except:
		print f + "failed"
		failed.append(f)


print "success " + str(len(succ))


print "failed " + str(len(failed))
	


