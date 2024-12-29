from mrjob.job import MRJob
from mrjob.step import MRStep
import logging
import numpy as np
from collections import Counter

# number of neighbors
K = 15

# Number of elements per Batch
N = 20


class MRKnn(MRJob):
    """
    Impl from: https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/BigData/A%20MapReduce-based%20k-Nearest%20Neighbor%20Approach%20for%20Big%20Data%20Classification-IEEE%20BigDataSE-2015.pdf
    K nearest neighbors together with their distance values are emitted to the reduce phase
    Reduce phase determines which are the final k nearest neighbors
    """

    def configure_args(self):
        super(MRKnn, self).configure_args()
        self.add_file_arg("--test", help="Path to the test file")
        logging.basicConfig(level=logging.INFO)

    def load_data(self, path):
        with open(path, "r") as f:
            return [
                line.split(",")
                for i, line in enumerate(f.read().splitlines())
                if i != 0
            ]

    def mapper_init(self):
        self.batch = [[], [], []]
        self.test_set = self.load_data(self.options.test)

    def mapper(self, _, line):
        """
        Each mapper receives 10 lines from the input file
        => Will compute the distance for 10 points for each of the 3 values in test
        Compute distance for each x in test_set with new line
        """
        y = line.split(",")

        # For each incoming train entry, need to compute the distance
        if y[0] != "Id":  # do not need header line
            y_class = y[-1]  # save the class
            y = np.array(y[1:-1], dtype=float)  # remove the ID from train set

            for i, x in enumerate(self.test_set):
                x = np.array(x[1:], dtype=float)  # do not need ID for distance
                distance = np.linalg.norm(y - x)
                self.batch[i].append((y_class, distance))

        if len(self.batch[0]) == 10:
            for i, x in enumerate(self.test_set):
                yield x[0], self.batch[i]  # yield each x by ID

    # yield last batch if n != 10
    def mapper_final(self):
        for i, x in enumerate(self.test_set):
            yield x[0], self.batch[i]  # yield each x by ID

    def reducer_init(self):
        self.neighbors = [(None, float("inf"))] * K

    def reducer(self, key, value):
        """
        Key is Id of test set
        Value is a list of type (class, distance)
        Will Need to keep the K closest distances
        """

        def replace_if_smaller(new_tuple, lst):
            """
            Changes inplace the maximum value if new_tuple is smaller
            """
            max_tuple = max(lst, key=lambda x: x[1])

            if new_tuple[1] < max_tuple[1]:
                max_index = lst.index(max_tuple)
                lst[max_index] = new_tuple

            return lst

        # double nested list, take first elem to remove outer list
        for batch in list(value):
            for elem in batch:
                self.neighbors = replace_if_smaller(tuple(elem), self.neighbors)
        counter = Counter(tp[0] for tp in self.neighbors)
        # 0 to select list, 0 to select first value of tuple
        most_common = counter.most_common(1)[0][0]
        yield "Solution: ", (key, most_common)

    def reducer_solution(self, key, value):
        """
        Used to aggregate the solution
        """
        yield key, list(value)

    def steps(self):
        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper,
                mapper_final=self.mapper_final,
                reducer_init=self.reducer_init,
                reducer=self.reducer,
            ),
            MRStep(
                reducer=self.reducer_solution
            )
        ]


if __name__ == "__main__":
    MRKnn.run()
