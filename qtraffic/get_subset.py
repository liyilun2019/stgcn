
def get_subset():
	subset=set();
	with open("qtraffic_roadSubset","r") as f:
		for line in f.readlines():
			subset.add(int(line))
	return subset
