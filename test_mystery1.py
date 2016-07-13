import homework9 as hw9
import unittest
import timeit
import random
import homework9_data as data
class TestHW9(unittest.TestCase):
    
    def test_mystery1_classifier(self):
        c = hw9.MysteryClassifier1(data.mystery1);
        self.assertEquals([False, False, False, True, True],[c.classify(x) for x in ((0, 0), (0, 1), (-1, 0), (1, 2), (-3, -4))]);
    
    def test_accuracy(self):   
        all_data = data.mystery1;  
        c = hw9.MysteryClassifier1(all_data);
        error = 0
        for test, y in all_data:
            if c.classify(test) != y:
                error += 1;

        accuracy = 1 - (error)/float(len(all_data));

        print "Mystery1:{}".format(accuracy);

    def test_cross_validation(self):
        all_data = data.mystery1
        folds = 10;
        subset_size = len(all_data) / folds;
        test_avg = 0;
        train_avg = 0;
        for i in xrange(folds):
            train = all_data[i * subset_size:][:subset_size]
            test = all_data[: i * subset_size] + all_data[(i+1) * subset_size:]
            c = hw9.MysteryClassifier1(train);
            error = 0;
            for x, y in test:
                y_hat = c.classify(x);
                if y_hat <> y:
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

