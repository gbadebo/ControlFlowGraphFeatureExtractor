from utils import buildCFG

cfg = buildCFG('testcode/example.c', 'addNumbers')

print "[+] Size of the CFG:", str(cfg.size())
print cfg.printer()
