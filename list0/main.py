import tsplib95


def read_example():
    att48 = tsplib95.load("../datasets/att48.opt.tour")
    print(att48.as_name_dict())


if __name__ == "__main__":
    read_example()
