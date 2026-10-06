import json,sys
nb=json.load(open(sys.argv[1],encoding='utf8'))
for c in nb['cells']:
    t=''.join(c['source'])
    if c['cell_type']=='code':
        if 'matplotlib' in t and 'FancyBbox' in t: continue
        print('#---code\n'+t)
    else: print('#---md\n'+t)
