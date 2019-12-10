class ConvexOptimizer:
    """Parent class for convex optimizers. 
    
    Attributes: specific to different optimizers
    Methods:
        optimize: perform optimization, return the final result. 
    """
    
    def __init__(self):
        raise NotImplementedError
        

    def optimize(self,x,grad,inspect=False):
        pass


class GradientDescent(ConvexOptimizer):
    """Gradient descent optimizer"""

    def __init__(self,learning_rate=1E-3,max_iteration=1000,batch=False):
        self.learning_rate=learning_rate
        self.max_iteration=max_iteration
        
        if batch is not False:
            raise NotImplementedError

    def optimize(self,x,grad,inspect=False):
        """Optimize by gradient descent.
        
        x: data 
        grad: gradient function at x
        """

        # TODO: change the converging criteria

        for i in range(self.max_iteration):
            x-=self.learning_rate*grad(x)
            if inspect and i % inspect==0:
                print("i=%d, x=%.5f" % (i,x)) 
            
        return x
    
