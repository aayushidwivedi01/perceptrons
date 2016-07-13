import homework9 as hw9
import unittest
import timeit
import random
import homework9_data as data
class TestHW9(unittest.TestCase):

    def test_predict_binary(self):
        train = [({"x1": 1}, True), ({"x2": 1}, True), ({"x1": -1}, False),\
                ({"x2": -1}, False)]
        test = [{"x1": 1}, {"x1": 1, "x2": 1}, {"x1": -1, "x2": 1.5},\
                {"x1": -0.5, "x2": -2}];
        p = hw9.BinaryPerceptron(train, 1)
        self.assertEquals([True, True, True, False] , [p.predict(x) for x in test])

    def test_predict_multiclass(self):
        train = [({"x1": 1}, 1), ({"x1": 1, "x2": 1}, 2), ({"x2": 1}, 3),\
        ({"x1": -1, "x2": 1}, 4), ({"x1": -1}, 5), ({"x1": -1, "x2": -1}, 6),\
        ({"x2": -1}, 7), ({"x1": 1, "x2": -1}, 8)];

        p = hw9.MulticlassPerceptron(train, 10);
        self.assertEquals([1, 2, 3, 4, 5, 6, 7, 8], [p.predict(x) for x, y in train])

    def test_iris_classifier(self):
        c = hw9.IrisClassifier(data.iris);
        self.assertEquals('iris-setosa' ,c.classify((5.1, 3.5, 1.4, 0.2)))

if __name__ == '__main__':
    unittest.main()

