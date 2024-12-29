from mrjob.job import MRJob
from mrjob.step import MRStep
import logging
import math


# Number of // reducers
K = 10


class MRKnn(MRJob):
    """
    Computes Frobenuis Norm Of Matrix
    1. Mapper converts each line from strings to arrays
    2. 1st reducer computes the sum of squared absolute value of each row
    3. 2nd reducer sums over all rows + sqrt
    """

    def configure_args(self):
        super(MRKnn, self).configure_args()
        logging.basicConfig(level=logging.INFO)

    def mapper(self, _, line):
        """
        Mapper converts to floats
        Need to provide a key for each mapper so that it is
        sent to reducers in //. If we provide None as a key all
        rows are going to be sent to the same reducer
        """
        row = list(map(float, line.split(" ")))
        key = hash(tuple(row)) % K
        yield key, row

    def reducer_1(self, key, value):
        """
        First reducer computes the sum for each row
        """
        for row in list(value):
            yield "sum", sum(list(map(lambda x: abs(x) ** 2, row)))

    def reducer_2(self, key, value):
        """
        Sums every incoming value and takes the sqrt
        """
        res = math.sqrt(sum(value))
        logging.info(f"Norm: {res}")
        yield "Norm", res

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer_1),
            MRStep(reducer=self.reducer_2),
        ]


if __name__ == "__main__":
    MRKnn.run()
