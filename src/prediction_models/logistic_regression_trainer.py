from global_types import EdgeList
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from dataset.data_splitting import construct_x_y
import optuna
from sklearn.metrics import average_precision_score
from numpy import ndarray
from typing import Tuple

class LogisticRegressionTrainer:
    __MIN_DIM = 80
    __MAX_DIM = 220
    __MIN_BETA = -1
    __MAX_BETA = 0
    __MIN_NUM_ITER_WEIGHTS = 3
    __MAX_NUM_ITER_WEIGHTS = 10
    __MIN_ITER_WEIGHT = -2.0
    __MAX_ITER_WEIGHT = 2.0
    __MIN_INV_L2_REG = 0.01
    __MAX_INV_L2_REG = 1

    def __init__(self, 
        seed: int,
        embeddings: ndarray,
        positive_train_edges: EdgeList, 
        negative_train_edges: EdgeList, 
        positive_validation_edges: EdgeList, 
        negative_validation_edges: EdgeList, 
        n_trials = 5
    ) -> None:
        self.__seed = seed
        self.__positive_train_edges = positive_train_edges
        self.__negative_train_edges = negative_train_edges
        self.__positive_validation_edges = positive_validation_edges
        self.__negative_validation_edges = negative_validation_edges
        self.__n_trials = n_trials
        self.__embeddings = embeddings

    def train_model(self) -> Tuple[LogisticRegression, float]:
        study = optuna.create_study(direction="maximize")
        study.optimize(self.__objective, n_trials=self.__n_trials)
        
        C = study.best_params["C"]

        clf = self.__train_model(C)

        return clf, study.best_trial.value

    def __train_model(
        self,
        C: float
    ) -> Tuple[LogisticRegression, ndarray]:
        X_train, Y_train = construct_x_y(
            self.__positive_train_edges, self.__negative_train_edges, self.__embeddings
        )

        X_train, Y_train = shuffle(X_train, Y_train, random_state=self.__seed)
        clf = LogisticRegression(max_iter=300, C=C, penalty="l2", random_state=self.__seed, solver="saga").fit(
            X_train, Y_train
        )

        return clf

    def __objective(
        self, 
        trial: optuna.Trial,
    ):
        # Set up trial parameters
        C = trial.suggest_float("C", self.__MIN_INV_L2_REG, self.__MAX_INV_L2_REG)

        # Train model
        clf = self.__train_model(C)

        # Test
        X_validation, Y_validation = construct_x_y(
            self.__positive_validation_edges, self.__negative_validation_edges, self.__embeddings
        )

        return average_precision_score(Y_validation, clf.predict(X_validation))