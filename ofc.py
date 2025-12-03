# TODO: change to numpy
# TODO: randomize connections

from collections import deque
from random import random

L = 40
ALPHA = 0.21
F_TH = 1


STEPS = 5
# set to random values initially
f: list[list[float]] = [[random() for _ in range(L)] for _ in range(L)]

adj: dict[tuple[int,int], list[int]] = dict()
for i in range(L):
    for j in range(L):
        adj[(i,j)] = []
        for (di,dj) in ((-1,0),(1,0),(0,-1),(0,1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < L and 0 <= nj < L:
                adj[(i,j)].append((ni,nj))

for step in range(STEPS):
    print(f"===== STEP {step} =====")



    active_queue = deque()

    for i in range(L):
        for j in range(L):
            f[i][j] = min(F_TH, f[i][j]+0.1)
            if f[i][j] >= F_TH:
                active_queue.append((i,j))

    earthquake_size = 0
    while len(active_queue) > 0:
        (ui, uj) = active_queue.popleft()
        fi = f[ui][uj]

        for (vi, vj) in adj[(ui,uj)]:
            if f[vi][vj] < F_TH:
                f[vi][vj] = min(f[vi][vj]+ALPHA*fi, F_TH)
                if f[vi][vj] == F_TH:
                    active_queue.append((vi,vj))
        earthquake_size += 1

    if earthquake_size > 0:
        print(f"EARTHQUAKE OF SIZE {earthquake_size} OCCURRED")

    for i in range(L):
        for j in range(L):
            if f[i][j] == F_TH:
                f[i][j] = 0
                print("#", end="")
            else:
                print(".", end="")
        print()



    
    


    
