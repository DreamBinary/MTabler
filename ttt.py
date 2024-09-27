import itertools
from collections import defaultdict

# 1到4全排列
orders4 = list(itertools.permutations([0, 1, 2, 3]))
orders8 = list(itertools.permutations([0, 1, 2, 3, 4, 5, 6, 7]))
ordered2 = defaultdict(lambda: defaultdict(list))
ordered3 = defaultdict(lambda: defaultdict(list))

print(len(orders4), len(orders8))

import random
ri = 5
arr = ["A", "B", "C", "D"]
label="ABCD"
order = orders4[ri]
print(order)
for i, j in enumerate(list(order)):
    print(f"{label[i]}) {arr[j]}")
# shuffled = [f"{label[i]}) {arr[j]}" for i, j in enumerate(list(order))]
# print(shuffled)
