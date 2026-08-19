for n in range(1,2):
    for m in range(1, 100):
        for o in range(1, 100):
            if n * m * o == target:
                found = n
                print(n,m,o,target)
