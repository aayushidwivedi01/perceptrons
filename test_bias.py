import homework9 as hw9
import unittest
import timeit
import random
import homework9_data as data
class TestHW9(unittest.TestCase):
    
    def test_bias_classifier(self):
        c = hw9.BiasClassifier(data.bias);
        self.assertEquals([False, False, False, True, True],[c.classify(x) for x in (-1, 0, 0.5, 1.5, 2)])
    
    def test_accuracy(self):   
        all_data = sorted([(x,y) for x,y in data.bias], key=lambda x:x[0], reverse=True);  
        c = hw9.BiasClassifier(all_data);
        print c.binary_perceptron.w
        error = 0
        for test, y in all_data:
            if c.classify(test) != y:
                error += 1;

        accuracy = 1 - (error)/float(len(all_data));

        print "BIAS:{}".format(accuracy);

    def test_cross_validation(self):
        all_data = data.bias;
        folds = 10;
        subset_size = len(all_data) / folds;
        test_avg = 0;
        train_avg = 0;
        for i in xrange(folds):
            train = all_data[i * subset_size:][:subset_size]
            test = all_data[: i * subset_size] + all_data[(i+1) * subset_size:]
            c = hw9.BiasClassifier(train);
            print c.binary_perceptron.w
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

