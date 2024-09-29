from model.setup import Model
from preprocess.setup import Preprocess
from visualization.setup import Visualization


class Pipeline:

    def __init__(self):
        self.__model = Model()
        self.__preprocess = Preprocess()
        self.__visualization = Visualization()
        
        self.__data = None

    def init(self, train: bool = False, data: list = []):
        if train:
            self.__model.train()
        else:
            self.__model.load()

        data = self.__preprocess.run(data)
        
        self.__data = data
        result = self.__model.predict(data)

        self.__visualization.run(data, result)

    def add(self, data):
        # here calculate probability to edges?
        self.__visualization.run(self.__data)

    def predict(self, data):
        # predict new symptoms/diseases etc
        self.__visualization.run(self.__data)
