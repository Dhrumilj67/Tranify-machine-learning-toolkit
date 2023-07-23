from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


models_dict = {
    "Logistic Regression": LogisticRegression(max_iter=100, random_state=100_000),
    "Decision Tree": DecisionTreeClassifier(random_state=100_000),
    "Random Forest": RandomForestClassifier(random_state=100_000),
    "SVM": SVC(random_state=100_000),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    # "XGBoost": XGBClassifier(random_state=100_000),
    "Neural Network": MLPClassifier(max_iter=50, random_state=100_000)
}
