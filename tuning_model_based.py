from surprise import SVD #Matrix factorization algorithm
from surprise import Dataset
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin("ml-100k")

param_grid  = {
    "n_epochs": [5,10], #Number of iterations to minimize function
    "lr_all": [0.002, 0.005], #leanring rate for parameters
    "reg_all": [0.4, 0.6] #regularization, penalty to prevent overfitting
}

gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])