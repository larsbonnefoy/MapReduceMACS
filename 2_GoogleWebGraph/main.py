from mrjob.job import MRJob


class ReverseGraph(MRJob):
    def mapper(self, _, line):
        """
        Invert target and source
        """
        if line[0] != "#":  # skip first lines in document
            source, target = line.split("\t")  # split on tab
            yield target, source

    def reducer(self, target, source):
        """
        Yields result
        """
        yield target, list(source)


if __name__ == "__main__":
    ReverseGraph.run()
