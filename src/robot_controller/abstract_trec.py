"""
abstract_trec.py

Contains abstract training recipe, an abstraction of the training. 
"""

from abc import ABC, abstractmethod


class AbstractTrainingRecipe(ABC):
    """A class that encapsulates the training/finetuning of one or more robot controllers trec-*. It describes the training data, validation data, the initial state, final state. It contains the training code (or wraps it). 
    
    The main aspects of the training recipe are:
    -it refers to the training of one or more rccos over some training data
    -the rccos start from a specific state, this might be the result of previous training
    -it might also be real time, eg RL (not yet clear how to implement this)
    -it not all the rcco-s might be changing
    -those that had been changed will be written out under a new state, new exp
    -the output exps contain a log of the training

    -there might be a relationship to a full roco, but it does not necessarily have to instantiated here
    -it should have support for snapshots, recover from snapshots etc

    """
    @abstractmethod
    def __init__(self, exp_trec):
        """Initialize the components of the trec, load the initial states"""
        raise NotImplementedError

    @abstractmethod
    def train(self, cont_train = True):
        raise NotImplementedError

    @abstractmethod
    def load(self, snapshot_no = 0):
        """Loads the state at the specified snapshot"""
        raise NotImplementedError
    
    @abstractmethod
    def save(self, new_snapshot = True):
        """Saves the state as a new snapshot"""
        raise NotImplementedError

    @abstractmethod
    def export(self):
        """Export the newly trained rcco-s"""
        raise NotImplementedError
