#set sample
example = set()

#dir(example)

example.add(42)
example.add(False)
example.add(3.14159)
example.add("Thorium")

print(example)

example2 = set([42,False,3.14159,"Thorium"])

print(len(example2))

odds = set([1, 3, 5, 7, 9])
evens = set([2, 4, 6, 8, 10])
primes = set([2, 3, 5, 7])
composites = set([4, 6, 8, 9, 10])

print(odds.union(evens))
print(evens.union(odds))

print(odds.intersection(primes))
print(primes.intersection(evens))
print(evens.intersection(odds))
print(primes.union(composites))

if 2 in primes:
    print("2 in primes is True")
else:
    print("2 in primes isFalse")


if 6 in odds:
    print("6 in odds is True")
else:
    print("6 in odds isFalse")