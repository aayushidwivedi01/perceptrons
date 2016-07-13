############################################################
# CIS 521: Homework 9
############################################################

student_name = "Aayushi Dwivedi"

############################################################
# Imports
############################################################

import homework9_data as data
from collections import defaultdict
import operator
# Include your imports here, if any are used.



############################################################
# Section 1: Perceptrons
############################################################

class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        w = defaultdict(float);

        for i in xrange(iterations):
            for (x, y) in examples:
                y_hat = True if sum([w[feature] * value for feature, value in x.iteritems()]) > 0 else False;
                if y_hat != y:
                    for (feature, value) in x.iteritems():
                        w[feature] += value * (1 if y else -1);
        self.w = w;

    def predict(self, x):
        if sum([ self.w[feature] * value for feature, value in x.iteritems()]) > 0:
            return True;
        else: 
            return False;

class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        self.w = {y:defaultdict(float) for x, y in examples};
        
        for i in xrange(iterations):
            for (x, y) in examples:
                y_hat = self.predict(x);
                if y_hat != y:
                    for feature, value in x.iteritems():
                        self.w[y][feature] += value;
                        self.w[y_hat][feature] -= value;

    def predict(self, x):
        results = defaultdict(float);
        for label, weights in self.w.iteritems():
            results[label] = sum([self.w[label][feature] * value for feature, value in x.iteritems()]);
        return max(results.iteritems(), key=operator.itemgetter(1))[0]

############################################################
# Section 2: Applications
############################################################

class IrisClassifier(object):

    def __init__(self, data):
        examples = [({i:x[i] for i in xrange(len(x))},y)for x, y in data]
        self.multiclass_perceptron = MulticlassPerceptron(examples, 10);
            

    def classify(self, instance):
        test = {i:instance[i] for i in xrange(len(instance))}
        return self.multiclass_perceptron.predict(test);

class DigitClassifier(object):

    def __init__(self, data):
        examples = [({i:x[i] for i in xrange(len(x))},y)for x, y in data]
        self.multiclass_perceptron = MulticlassPerceptron(examples, 10);


    def classify(self, instance):
        test = {i:instance[i] for i in xrange(len(instance))}
        return self.multiclass_perceptron.predict(test);

class BiasClassifier(object):

    def __init__(self, data):
        self.avg = sum(x for x, y in data)/len(data);
        examples = [({0:x, 1:self.avg},y) for x, y in data]          
        self.binary_perceptron = BinaryPerceptron(examples, 10);
        

    def classify(self, instance):
        test = {0:instance, 1:self.avg}
        return self.binary_perceptron.predict(test);  

class MysteryClassifier1(object):

    def __init__(self, data):
        examples = [((x[0]**2 + x[1]**2), y) for x, y in data]
        self.binary_perceptron = BiasClassifier(examples);


    def classify(self, instance):
        test = (instance[0]**2 + instance[1]**2) 
        return self.binary_perceptron.classify(test);


class MysteryClassifier2(object):

    def __init__(self, data):
        examples = [({0:x*y*z},l) for ((x,y,z),l) in data ]
        self.binary_perceptron = BinaryPerceptron(examples,1);

    def classify(self, instance):
        x = instance[0]; y = instance[1]; z = instance[2]
        test = {0:x*y*z}
        return self.binary_perceptron.predict(test);

############################################################
# Section 3: Feedback
############################################################

feedback_question_1 = """
8
"""

feedback_question_2 = """
Finding features for bias and mystery classifier 2.
No.
"""

feedback_question_3 = """
I liked the fact that I had to experiment with different feature.
I would have liked it if MysteryClassifier2 had required some
geometric transformation. Also, some dev data was provided.
"""
