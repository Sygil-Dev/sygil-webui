from util.imports import *

def seed_to_int(s):
	if type(s) is int:
		return s
	if s is None or s == '':
		return random.randint(0, 2**32 - 1)
	n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
	while n >= 2**32:
		n = n >> 32
	return n
