import tsplib95


def read_example():
  dantzing42 = tsplib95.load('../datasets/dantzig42.tsp')
  print(dantzing42.as_name_dict())

if __name__ == "__main__":
  read_example()