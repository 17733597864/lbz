import joblib

# 假设这个单元格的代码不是在这台电脑上，而是在运行模型的服务器
# 下面代码省略完整的

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    # tarball_path = Path("datasets/housing.tgz")
    # if not tarball_path.is_file():
    #     Path("datasets").mkdir(parents=True, exist_ok=True)
    #     url = "https://github.com/ageron/data/raw/main/housing.tgz"
    #     urllib.request.urlretrieve(url, tarball_path)
    # with tarfile.open(tarball_path) as housing_tarball:
    #         housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("D:\ml-teach-main(1)\ml-teach-main\datasets\housing\housing.csv"))

housing = load_housing_data()

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

#class ClusterSimilarity(BaseEstimator, TransformerMixin):
#    [...]

final_model_reloaded = joblib.load("D:\ml-teach-main(1)\ml-teach-main\my_california_housing_model.pkl")

new_data = housing.iloc[:5]  # pretend these are new districts
predictions = final_model_reloaded.predict(new_data)
