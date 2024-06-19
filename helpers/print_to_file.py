def print_to_file(fname, data):
    """Print list to text file"""
    with open(fname, "w", encoding="utf-16") as file:
        for item in data:
            file.write("%s\n" % item)
