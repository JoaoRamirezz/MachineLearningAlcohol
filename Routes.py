from flask import Flask, request
from joblib import load

app = Flask(__name__)

@app.route('/DecisionTree', methods=['POST'])
def DecisionPredict():
    data = request.get_json()
    data = data['key']
    data = data.split(",")
    data = [data]
    mymodel = load("./models/mymodelDecisionTree.pkl")
    result = str(mymodel.predict(data)[0])
    
    match result:
        case "1":
            return "very low" 
        case "2":
            return "low" 
        case "3":
            return "medium" 
        case "4":
            return "high" 
        case "5":
            return "very high" 


@app.route('/LinearRegression', methods=['POST'])
def LinearPredict():
    data = request.get_json()
    data = data['key']
    data = data.split(",")
    data = [data]
    
    mymodel = load("./models/mymodelLinearRegression.pkl")
    result = str(mymodel.predict(data)[0])
    
    match result:
        case "1":
            return "very low" 
        case "2":
            return "low" 
        case "3":
            return "medium" 
        case "4":
            return "high" 
        case "5":
            return "very high" 
        

@app.route('/KNeighbors', methods=['POST'])
def KNeighborsPredict():
    data = request.get_json()
    data = data['key']
    data = data.split(",")
    data = [data]
    
    mymodel = load("./models/mymodelKnn.pkl")
    result = str(mymodel.predict(data)[0])
    
    match result:
        case "1":
            return "very low" 
        case "2":
            return "low" 
        case "3":
            return "medium" 
        case "4":
            return "high" 
        case "5":
            return "very high" 



app.run(port=8080)