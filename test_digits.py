import homework9 as hw9
import unittest
import timeit
import random
import homework9_data as data
class TestHW9(unittest.TestCase):
    
    def test_digit_classifier(self):
        c = hw9.DigitClassifier(data.digits);
        self.assertEquals(0, c.classify((0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0)));
    
    def test_accuracy(self):   
        all_data = data.digits;  
        c = hw9.DigitClassifier(all_data);
        error = 0
        for test, y in all_data:
            if c.classify(test) != y:
                error += 1;

        accuracy = 1 - (error)/float(len(all_data));

        print "DIGITS:{}".format(accuracy);

    def test_cross_validation(self):
        all_data = data.digits
        folds = 10;
        subset_size = len(all_data) / folds;
        test_avg = 0;
        train_avg = 0;
        for i in xrange(folds):
            train = all_data[i * subset_size:][:subset_size]
            test = all_data[: i * subset_size] + all_data[(i+1) * subset_size:]
            c = hw9.DigitClassifier(train);
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

