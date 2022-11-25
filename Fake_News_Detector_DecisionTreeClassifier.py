from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import export_graphviz
import numpy as np
import graphviz
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def load_data():
    """
    Load the feature and target dataset of real and fake news headlines
    Convert the data to a matrix of TF-IDF features
    Returns:
        data (Dict): contains the training, validation and test dataset
        features (array shape (5799, )): contains strings of all different words in the dataset
    """
    real, real_y = csv_to_list("clean_real.txt")
    fake, fake_y = csv_to_list("clean_fake.txt")
    # Merge the datasets of headlines
    headlines = real + fake
    # Create a numpy array contraining the target value of each example
    y = np.concatenate((np.ones(real_y), np.zeros(fake_y)))
    # Create Vectorizer
    #
    vec = TfidfVectorizer()
    # The fit method is calculating the mean and variance of each feature presents in our data.
    # The transform method is transforming all the features using the respective mean and variance.
    # headlines is np array of shape (3622, 5799) where 3622 is the nnumber of headlines in the dataset and
    # 5799 is the number of different word in the dataset
    headlines = vec.fit_transform(headlines)
    data = split_data(headlines, y)
    features = vec.get_feature_names_out()
    return data, features

def csv_to_list(path):
    """
    Load csv file into list of strings
    Args:
        path (String): file path of dataset
    Returns:
        lines (List): Contains strings of headlines
        count (int): Number of lines in the dataset
    """
    file = open(path, 'r')
    lines = []
    for l in file:
        lines.append(l.strip())
    return lines, len(lines)

def split_data(X, y, train_size = 0.7):
    """
    Split the datasets into different sets
    Args:
        X (ndarray): feature value of each example
        y (ndarray): target value of each example
        train_size (float): proportion of the dataset to include in the training set
    Returns:
        data (Dict): contains the training, validation and test dataset
    """
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.5)
    res = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    return res

def select_model(data, depth):
    """
    Find the models with the hyperparameters criterion and max_depth that achieved the highest validation accuracy
    Args:
        data (Dict): contains the training, validation and test dataset
        depth (List[int]): contains different depth of the decision tree
    Returns:
        model (List[Object]): two models with best validation accuracy based on gini and entropy criteria
    """
    model = []
    for criteria in ['gini', 'entropy']:
        print("The criteria is {}".format(criteria))
        best_depth = 0
        best_acc = 0
        best_model = None
        for d in depth:
            clf = DecisionTreeClassifier(criterion=criteria, max_depth=d, random_state=1234)
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
    

np.random.seed(5555)
data, feature_names = load_data()
depth = [5,10,20,35,50]
model_gini, model_entropy = select_model(data, depth)
# visulization(model_gini, feature_names, "gini")
# visulization(model_gini, feature_names, "entropy")


# Try XGBClassifier
print("Try XGBClassifier")
xg_model = XGBClassifier(eval_metric="rmse", use_label_encoder=False)
xg_model.fit(data['train'][0], data['train'][1])

# *data['train'] unpacks the tuple, equiv to write data['test'][0], data['test'][1]
print("Train_acc: " + str(xg_model.score(*data['train'])))
# y_pred = xg_model.predict(data['val'][0])
# print(accuracy_score(data['val'][1], y_pred)*100)
print("Val_acc: " + str(xg_model.score(*data['val'])))
print("Test_acc: " + str(xg_model.score(*data['test'])))
