from collections import defaultdict

d = defaultdict(lambda: defaultdict(int))


d[1]["a"] += 1
print(d)
