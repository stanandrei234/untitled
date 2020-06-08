import numpy as np

def L(X,W,B):

    return 1/(1+np.exp(
        (-1)*(
            X*W.T+B
        )
    ))

def cost_function(Y,YY):
    #number of elements
    n=len(Y)

    #calculate the cost function between actual and predicted
    return  1/n*sum(Y*np.log(YY)+(1-Y)*np.log(1-YY))



def predict(X,Y,W,B,iteratii, learning_rate):

    pass





