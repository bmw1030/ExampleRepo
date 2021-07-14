from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load and split the data

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size= 0.2,random_state=42 )
X_train.shape

print(X_train)

pipe_lr = Pipeline([('minmax', MinMaxScaler()),
 ('lr', LogisticRegression())])
pipe_lr.fit(X_train, y_train)

score = pipe_lr.score(X_test, y_test)
print('Logistic Regression pipeline test accuracy: %.3f' % score)