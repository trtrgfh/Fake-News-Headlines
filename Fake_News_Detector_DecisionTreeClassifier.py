from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import export_graphviz
import numpy as np
import graphviz
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def load_data():
    """
    Load data from csv of real news headlines and fake new headlines
    Convert the data to a matrix of TF-IDF features.
    Then return the splited data and the features in the data
    """
    real, real_y = csv_to_list("/Fake News Detector/clean_real.txt")
    fake, fake_y = csv_to_list("/Fake News Detector/clean_fake.txt")
    headlines = real + fake
    y = np.concatenate((np.zeros(real_y), np.ones(fake_y)))
    vec = TfidfVectorizer()
    # The fit method is calculating the mean and variance of each of the features present in our data.
    # The transform method is transforming all the features using the respective mean and variance.
    headlines = vec.fit_transform(headlines)
    return split_data(headlines, y), vec.get_feature_names()

def csv_to_list(path):
    """
    Return a list of strings containing the lines in a file, and the number of lines in the file
    """
    file = open(path, 'r')
    lines = []
    count = 0
    for l in file:
        line = [l.strip()]
        lines.extend(line)
        count += 1
    return lines, count

def split_data(X, y, train_size = 0.7, val_size = 0.15):
    """
    Return a dictionary containing the traning set, validation set and the test set
    """
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.5)
    res = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    return res

def select_model(data, depth):
    """
    Return the models with the hyperparameters criterion and max_depth that achieved the highest validation accuracy
    """
    model = []
    for criteria in ['gini', 'entropy']:
        print("The criteria is {}".format(criteria))
        best_depth = 0
        best_acc = 0
        best_model = None
        for d in depth:
            clf = DecisionTreeClassifier(criterion=criteria, max_depth=d)
            clf.fit(data["train"][0], data["train"][1])
            # Return mean accuracy of self.predict(X) wrt. y.
            val_acc = clf.score(data["val"][0], data["val"][1])
            train_acc = clf.score(data["train"][0], data["train"][1])
            print("Depth: {}, Train_acc: {}, Val_acc: {}".format(d, train_acc, val_acc))
            if val_acc > best_acc:
                best_depth = d
                best_acc = val_acc
                best_model = clf
        print("Best Depth: {}, Test_acc: {}".format(best_depth, best_model.score(*data["test"])))
        model.append(best_model)
    return model

def visulization(model, feature_names, criterion):
    dot_data = export_graphviz(model, feature_names=feature_names, filled=True)
    graph = graphviz.Source(dot_data)
    graph.render('graph_{}'.format(criterion))


data, feature_names = load_data()
depth = [1,5,10,20,35,50,75]
model_gini, model_entropy = select_model(data, depth)
visulization(model_gini, feature_names, "gini")
visulization(model_gini, feature_names, "entropy")



# Try XGBClassifier
from sklearn.model_selection import GridSearchCV
# xg_model = XGBClassifier()
# grid_values = {"learning_rate"    : [0.01, 0.05, 0.1, 0.15, 0.3] ,
#  "max_depth"        : [ 1, 5, 10, 20, 35, 50, 75]}
 # "min_child_weight" : [ 1, 3, 5, 7 ],
 # "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 # "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
# grid_search = GridSearchCV(xg_model, param_grid = grid_values, scoring="accuracy")
# grid_result = grid_search.fit(data['train'][0], data['train'][1])
print("XGBClassifier")
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
xg_model = XGBClassifier()
xg_model.fit(data['train'][0], data['train'][1])
y_pred = xg_model.predict(data['val'][0])
print(xg_model.score(*data['train']))
print(accuracy_score(data['val'][1], y_pred)*100)
print(xg_model.score(*data['test']))
