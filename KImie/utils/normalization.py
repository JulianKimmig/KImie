import numpy as np
from KImie import KIMIE_LOGGER

class Normalization():
    def __init__(self,):
        self._fitted=False

    def normalize(self,x):
        x=np.array(x).astype(float)
        if not self._fitted:
            raise ValueError("Normalization not fitted")
        if x.ndim==1:
            x=x.reshape(-1,1)
        if x.ndim!=2:
            raise ValueError("x must be 1D or 2D")
        return x
    
    def inverse(self,x):
        x=np.array(x).astype(float)
        if not self._fitted:
            raise ValueError("Normalization not fitted")
        if x.ndim==1:
            x=x.reshape(-1,1)
        if x.ndim!=2:
            raise ValueError("x must be 1D or 2D")
        return x

    def fit(self,x):
        x=np.array(x).astype(float)
        if x.ndim==1:
            x=x.reshape(-1,1)
        if x.ndim!=2:
            raise ValueError("x must be 1D or 2D")
        self._fitted=True
        return x

def gen_ecdf(data):
    values,counts=np.unique(data,return_counts=True)
    ecdf=np.cumsum(counts)/np.sum(counts)
    if ecdf.shape[0]==1:
        ecdf=np.array([0.,1.])
        values=np.array([values[0]-1.,values[0]])

    ecdf-=ecdf[0]
    ecdf/=ecdf[-1]
    return values,ecdf

class ECDFNormalization(Normalization):
    def __init__(self,max_resolution:int=100,error_threshold:float=1e-5):
        super().__init__()
        self._max_resolution = max_resolution
        self._error_threshold = error_threshold
        self._dimensions=0
    
    def interpolation_points(self):
        return self._x_use,self._y_use
   # def inverse_interpolation_points(self):
   #     return self._ix_use,self._iy_use

    def fit(self,x):
        x=super().fit(x)
        self._dimensions=x.shape[1]
        self._x_use=[None]*self._dimensions
        self._y_use=[None]*self._dimensions
        ecdf_values=[]
        for d in range(self._dimensions):
            values,ecdf=gen_ecdf(x[:,d])
            ecdf_values.append((values,ecdf))
            x_use=np.array([values[0],values[-1]]+[values[-1]]*(self._max_resolution-2))
            y_use=np.array([ecdf[0],ecdf[-1]]+[1.0]*(self._max_resolution-2))
            ignore_mask=np.zeros_like(values,dtype=bool)
            ignore_mask[values==x_use[0]]=True
            ignore_mask[values==x_use[1]]=True
            data_dist=values.max()-values.min()
            
            for i in range(2,max(3,self._max_resolution)):
                if np.all(ignore_mask):
                    if (self._max_resolution-i)>0:
                        x_use=x_use[:-(self._max_resolution-i)]
                        y_use=y_use[:-(self._max_resolution-i)]
                    break
                error1=(np.interp(values[~ignore_mask],x_use,y_use)-ecdf[~ignore_mask])**2
                error2=((np.interp(ecdf[~ignore_mask],y_use,x_use)-values[~ignore_mask])/data_dist)**2
                error=(error1+error2)/2
                me=np.sqrt(np.mean(error))

                if me<self._error_threshold:
                    if (self._max_resolution-i)>0:
                        x_use=x_use[:-(self._max_resolution-i)]
                        y_use=y_use[:-(self._max_resolution-i)]
                    break

                x_use[i]=values[~ignore_mask][error.argmax()]
                y_use[i]=ecdf[~ignore_mask][error.argmax()]
                ignore_mask[values==x_use[i]]=True
                s=np.argsort(x_use)
                x_use=x_use[s]
                y_use=y_use[s]

            if np.all(ignore_mask):
                me = 0
            else:
                error1=(np.interp(values[~ignore_mask],x_use,y_use)-ecdf[~ignore_mask])**2
                error2=((np.interp(ecdf[~ignore_mask],y_use,x_use)-values[~ignore_mask])/data_dist)**2
                error=(error1+error2)/2
                me=np.sqrt(np.mean(error))
                if np.isnan(me):
                    raise ValueError(f"ECDF normalization failed ({data_dist},{error1},{error2},{error},{me})")
            KIMIE_LOGGER.info(f"normalization of feature {d} fitted with {x_use.shape[0]} points and an error of {me}")
            self._x_use[d]=x_use
            self._y_use[d]=y_use


    def normalize(self,x):
        x=super().normalize(x)
        if x.shape[1]!=self._dimensions:
            raise ValueError(f"Dimension mismatch (got {x.shape[1]}, expected {self._dimensions})")
        out=np.zeros_like(x)
        for d in range(self._dimensions):
            out[:,d]=np.interp(x[:,d],self._x_use[d],self._y_use[d])
        return out        

    def inverse(self,x):
        x=super().inverse(x)
        if x.shape[1]!=self._dimensions:
            raise ValueError(f"Dimension mismatch (got {x.shape[1]}, expected {self._dimensions})")
        out=np.zeros_like(x)
        for d in range(self._dimensions):
            out[:,d]=np.interp(x[:,d],self._y_use[d],self._x_use[d])
        return out
        
