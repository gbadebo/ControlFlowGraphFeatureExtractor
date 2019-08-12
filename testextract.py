from utils import buildCFG

#cfgs = buildCFG('Using_freed_memory.c')
filename = 'testcode.c'
cfgs = buildCFG(filename)
print cfgs
l = []
for cfg in cfgs:
	print cfg
	print "\n[+] Function: ", cfg[0]
	print cfg[1].printer()
	with open(filename, 'r') as content_file:
		content = content_file.read()
 	start = content.index(cfg[0])
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
	print funcContent
			
	
	
		
    
     

    
