# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:06:12 2017

@author: Andrea Del Prete

"""
import numpy as np
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds);

    def __str__(self, prefix=""):
        res = "";
        for (key,value) in self.__dict__.iteritems():
            if (isinstance(value, np.ndarray) and len(value.shape)==2 and value.shape[0]>value.shape[1]):
                res += prefix+" - " + key + ": " + str(value.T) + "\n";
            elif (isinstance(value, Bunch)):
                res += prefix+" - " + key + ":\n" + value.__str__(prefix+"    ") + "\n";
            else:
                res += prefix+" - " + key + ": " + str(value) + "\n";
        return res[:-1];