import homework9 as hw9
import unittest
import timeit
import random
import homework9_data as data
class TestHW9(unittest.TestCase):
    
    def test_iris_classifier(self):
        c = hw9.IrisClassifier(data.iris);
        self.assertEquals('iris-setosa' ,c.classify((5.1, 3.5, 1.4, 0.2)))

    def test_accuracy(self):     
        c = hw9.IrisClassifier(data.iris);
        error = 0
        for test, y in data.iris:
            if c.classify(test) != y:
                error += 1;

        accuracy = 1 - (error)/float(len(data.iris));

        print "IRIS:{}".format(accuracy);

    def test_cross_validation(self):
        folds = 10;
        subset_size = len(data.iris) / folds;
        test_avg = 0;
        train_avg = 0;
        for i in xrange(folds):
            train = data.iris[i * subset_size:][:subset_size]
            test = data.iris[: i * subset_size] + data.iris[(i+1) * subset_size:]
            c = hw9.IrisClassifier(train);
            error = 0;
            for x, y in test:
                if c.classify(x) <> y:
                    error += 1;                
            test_avg += 1 - (error)/float(len(test));
            
            error = 0;
            for x, y in train:
                if c.classify(x) <>y :
                    error += 1;
                
            train_avg += 1 - (error)/float(len(train));

        print "Test Avg:{}".format(test_avg/10);
        print "Train Avg:{}".format(train_avg/10);

if __name__ == '__main__':
    unittest.main()

