from utils import buildCFG

cfgs = buildCFG('Using_freed_memory.c')
print cfgs

for cfg in cfgs:
    print cfg
    print "\n[+] Function: ", cfg[0]
    print cfg[1].printer()


