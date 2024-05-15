#####################################################
import pandas as pd
from sklearn import linear_model
import os

PROJECT_PATH = os.path.abspath(os.path.dirname(__name__))
#####################################################

#####################################################
# Loading Dataset Globally
data = pd.read_csv(PROJECT_PATH+"/dataset/dataset.csv")
array = data.values

for i in range(len(array)):
    if array[i][0] == "Male":
        array[i][0] = 1
    else:
        array[i][0] = 0

df = pd.DataFrame(array)

maindf = df[[0, 1, 2, 3, 4, 5, 6]]
mainarray = maindf.values

temp = df[7]
train_y = temp.values
train_y = temp.values

for i in range(len(train_y)):
    train_y[i] = str(train_y[i])

mul_lr = linear_model.LogisticRegression(
    multi_class="multinomial", solver="newton-cg", max_iter=1000
)
mul_lr.fit(mainarray, train_y)
#####################################################

def predict(gender,age,openness,neuroticism,conscientiousness,agreeableness,extraversion):

    if age < 17:
        age = 17
    elif age > 28:
        age = 28

    inputdata = [
        [
            gender,
            age,
            9 - openness,
            9 - neuroticism,
            9 - conscientiousness,
            9 - agreeableness,
            9 - extraversion,
        ]
    ]

    print("Input:",inputdata)

    df1 = pd.DataFrame(inputdata)
    testdf = df1[[0, 1, 2, 3, 4, 5, 6]]
    maintestarray = testdf.values
    print(maintestarray)
    y_pred = mul_lr.predict(maintestarray)
    for i in range(len(y_pred)):
        y_pred[i] = str((y_pred[i]))
    DF = pd.DataFrame(y_pred, columns=["Predicted Personality"])
    DF.index = DF.index + 1
    DF.index.names = ["Person No"]
    result=DF["Predicted Personality"].tolist()[0]
    print("Result:",result)
    return result