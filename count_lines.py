count = 0
with open('./script/interfase.py') as fp:
    for line in fp:
        if line.strip():
            count += 1
print('number of non-blank lines', count)