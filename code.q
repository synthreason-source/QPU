target = 100
for n in range(10):
    for m in range(1, 10):
        for o in range(1, 10):
            if n * m * o == target:
                found = n
                print(n,m,o,target)
