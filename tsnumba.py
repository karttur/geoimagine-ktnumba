'''
Created on 19 Oct 2018

@author: thomasgumbricht
'''
import numba
import numpy as np
from math import sqrt
from scipy import stats, interpolate
from sympy.physics.quantum.gate import zx_basis_transform

def MKtestOld(x):  
    n = len(x)
    s = 0
    for k in range(n-1):
        t = x[k+1:]
        u = t - x[k]
        sx = np.sign(u)
        s += sx.sum()
    unique_x = np.unique(x)
    #print 'unique_x',unique_x
    g = len(unique_x)
    if n == g: 
        var_s = (n*(n-1)*(2*n+5))/18
    else:
        tp = np.unique(x, return_counts=True)[1]
        #print 'tp',tp
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18
    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    return z

@numba.jit(nopython=True)
def MKtestNumba(x,unique_x,dotp,tp):  
    n = len(x)
    s = 0
    for k in range(n-1):
        t = x[k+1:]
        u = t - x[k]
        sx = np.sign(u)
        s += sx.sum()
    if dotp:
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18   
    else:  
        var_s = (n*(n-1)*(2*n+5))/18
    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
            z = 0
    else:
        z = (s + 1)/np.sqrt(var_s)
    return z

def MKtestIni(x): 
    unique_x = np.unique(x)
    tp = np.array([0,1])
    dotp = 0
    if len(x) > len(unique_x):
        tp = np.unique(x, return_counts=True)[1]
        dotp = 1
    #print 'tp',tp
    return MKtestNumba(x,unique_x,dotp,tp) 

@numba.jit(nopython=True)
def ResampleToPeriodAvg(arr, y, step, nrperiods):
    for c in range(nrperiods):
        y[c] = arr[step*c:(c+1)*step].mean()       
    return y

@numba.jit(nopython=True)
def ToPeriodStd(x, y, step, nrperiods):
    for c in range(nrperiods):
        y[c] = x[step*c:(c+1)*step].std()       
    return y

@numba.jit(nopython=True)
def ToPeriodMin(x, y, step, nrperiods):
    for c in range(nrperiods):
        y[c] = x[step*c:(c+1)*step].min()       
    return y

@numba.jit(nopython=True)
def ToPeriodMax(x, y, step, nrperiods):
    for c in range(nrperiods):
        y[c] = x[step*c:(c+1)*step].max()       
    return y

@numba.jit(nopython=True)
def ToPeriodMulti(x, y, step, nrperiods):
    for c in range(nrperiods):
        ts = x[step*c:(c+1)*step]
        y[0][c] = ts.mean()   
        y[1][c] = ts.std() 
        y[2][c] = ts.min()   
        y[3][c] = ts.max()  
    return y

@numba.jit(nopython=True)
def ToAnnualSum(x, y, step, years):
    for c in range(years):
        y[c] = x[step*c:(c+1)*step].sum()       
    return y

@numba.jit(nopython=True)
def ResampleToSum(x, dstArr, dstSize, periods):
    for c in range(dstSize):
        dstArr[c] = x[periods*c:(c+1)*periods].sum()       
    return dstArr

@numba.jit(nopython=True)
def ResampleSeasonalAvg(x, dstArr, seasons):
    for c in range(seasons):
        seasonSum = x[c::seasons]
        season = np.nansum(seasonSum)
        n = seasonSum.shape[0] - np.isnan(seasonSum).sum()
        if n == 0:
            dstArr[c] = np.NaN
        else: 
            dstArr[c] = season/n  
    return dstArr

@numba.jit(nopython=True)
def ExtractMinMax(x):
    return (np.argmin(x),np.argmax(x))


@numba.jit(nopython=True)
def Zscore(x, y):
    return  (x-x.mean())/x.std()

@numba.jit(nopython=True)
def OLSregr(xyArr):  
    N = len(xyArr)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for i in range(len( xyArr )):
        Sx = Sx + xyArr[i][0]
        Sy = Sy + xyArr[i][1]
        Sxx = Sxx + xyArr[i][0]*xyArr[i][0]
        Syy = Syy + xyArr[i][1]*xyArr[i][1]
        Sxy = Sxy + xyArr[i][0]*xyArr[i][1]
    det = Sxx * N - Sx * Sx
    a, b = (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det
    meanerror = residual = bias = 0.0
    for i in range(len( xyArr )):
        meanerror = meanerror + (xyArr[i][1] - Sy/N)**2
        residual = residual + (xyArr[i][1] - a * xyArr[i][0] - b)*(xyArr[i][1] - a * xyArr[i][0] - b)
        bias += (xyArr[i][1] - (a * xyArr[i][0] + b))
    bias /= N
    RR = 1 - residual/meanerror
    residual = sqrt(residual/N)
    return (a,b,RR,residual,meanerror,bias)

@numba.jit(nopython=True)
def Rmse_metricNumba(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

def SsNumba(x):
    """Return sum of square deviations of sequence data."""
    c = np.mean(x)
    ss = sum((x-c)**2 for x in x)
    return ss

@numba.jit(nopython=True)
def CorrCoeffNumba(x, y):
    n = len(x)
    x_mean = np.mean(x)
    x_ss = np.sum((x_mean-x)*(x_mean-x))
    x_std = sqrt(x_ss/(n-1))
    y_mean = np.mean(y)
    y_ss = np.sum((y_mean-y)*(y_mean-y))
    y_std = sqrt(y_ss/(n-1))
    zx = (x-x_mean)/x_std
    zy = (y-np.mean(y))/y_std
    r = np.sum(zx*zy)/(len(x)-1)
    return r**2

@numba.jit(nopython=True)
def CovarianceNumba(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar
 
@numba.jit(nopython=True)
def VarianceNumba(values, mean):
    #return np.sum([(x-mean)**2 for x in values])
    var = 0
    for x in values:
        var += (x-mean)**2
    return var
        
@numba.jit(nopython=True)
def OLSextendedNumba(x,y,olsArr):
    n = len(x)
    x_mean = np.sum(x) / float(len(x))
    y_mean = np.sum(y) / float(len(y))
    #get variance
    var0 = VarianceNumba(x, x_mean)
    if var0 == 0:
        olsArr[0] = 0
        olsArr[1] = y_mean
        olsArr[2] = 1.0
        olsArr[3] = 0
    else:   
        #get covariance
        cov0 = CovarianceNumba(x, x_mean, y, y_mean)
        slope = cov0/var0
        olsArr[0] = slope
        #derive intercept
        intercept = y_mean - slope * x_mean
        olsArr[1] = intercept
        #get x sum of squares
        x_ss = np.sum((x_mean-x)*(x_mean-x))
        y_ss = np.sum((y_mean-y)*(y_mean-y))
        #get standard deviation   
        x_std = sqrt(x_ss/(n-1))
        y_std = sqrt(y_ss/(n-1))
        #calculate correlation coefficient
        zx = (x-x_mean)/x_std
        zy = (y-np.mean(y))/y_std
        r = np.sum(zx*zy)/(len(x)-1)
        r2 = r**2
        olsArr[2] = r2
        #r2 = CorrCoeffNumba(x,y)
        predict = intercept + slope * x
        rmse = Rmse_metricNumba(y,predict)
        olsArr[3] = rmse
    return olsArr

@numba.jit(nopython=True)
def InterpolateLinearNaNNumba(arr):
    #lastitem = arr.shape[0]
    for i in range(1,arr.shape[0]):  
        if np.isnan(arr[i]):   
            postIndexArray = arr[i+1:]
            postIndex = np.where(~np.isnan(postIndexArray))
            arr[i] = (arr[i-1]+(postIndexArray[postIndex[0][0]]/(postIndex[0][0]+1.0) )) / ( 1.0+ (1.0/(postIndex[0][0]+1.0) ) )
    return arr

@numba.jit(nopython=True)
def InterpolateLinearNumba(y,x,steps,filled):
    for item in range(1,len(x)):
        delta = float(y[item]-y[item-1]) / (x[item]-x[item-1])
        #interp_vals = [ y[item-1] + delta * z for z in range(steps[item-1]) ]
        #filled[x[item-1]:x[item]] = interp_vals
        filled[x[item-1]:x[item]] = [ y[item-1] + delta * z for z in range(steps[item-1]) ]
        #print item, 'y[item]',y[item], 'x[item]', x[item]
    #set the last value
    filled[x[item]] = y[item]
    return filled

@numba.jit(nopython=True)
def InterpolateLinearSeasonsNaN(arr,seasonArr,offset,seasons):
    lastitem = arr.shape[0]
    for i in range(1,arr.shape[0]):  
        if np.isnan(arr[i]):   
            postIndexArray = arr[i+1:lastitem]
            postIndex = np.where(~np.isnan(postIndexArray))
            weight = 2.8/(postIndex[0][0]**2+3.0)
            w = 1-weight
            s = i+offset+seasons
            y = int(s/seasons)
            t = i+seasons-(seasons*y)
            arr[i] = (arr[i-1]+ (postIndexArray[postIndex[0][0]]*weight) + (seasonArr[t]*w)) / 2.0 
    return arr

''' 
Nnanmean not supported at time of writing
@numba.jit(nopython=True)
def AverageTSPerDateNumba(arr, dstNull):
    Rr = np.nanmean( arr, axis=2 )
    Rr[np.isnan(Rr)] = dstNull
    return Rr
'''

def mean(values):
    return sum(values) / float(len(values))
 
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar
 
def variance(values, mean):
    return sum([(x-mean)**2 for x in values])
 
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

def interpolNan2(A):
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    
    A[np.isnan(A)] = np.interp(x, xp, fp)
    return A

@numba.jit(nopython=True)
def interpolNan(A):
    not_nan = np.logical_not(np.isnan(A))
    indices = np.arange(len(A))
    return np.interp(indices, indices[not_nan], A[not_nan])

@numba.jit(nopython=True)
def ResampleFixedPeriods(A,indexA,resultA):
    '''A is an array with daily values
    indexA is an array with the number of days in each consequtive month 
    resultA is the destination array
    '''
    for m in range(1,indexA.shape[0]):
        start = indexA[m-1]
        end = indexA[m]
        resultA[m-1] = A[start:end].mean()
    return resultA
        
'''    
x = np.array([0,10,20,30,50,60,99,61])   
print MKtestOld(x)
print MKtest(x)
#print MKtestReal(x)

x = np.array([0,10,20,30,50,60,99])
steps = [j-i for i, j in zip(x[:-1], x[1:])]
filled = np.arange( (x.max()- x.min()+1) )
y = np.array([2,4,5,8,4,5,3])

print InterpolateLinearNumba(y,x,steps,filled)


y = np.array([1, 1, 1, np.NaN, np.NaN, 2, 2, np.NaN, 0])

#y = nan_helper(y)

y = interpolNan(y)
print y.round(2)


y = InterpolateLinearNaNNumba(y)

print y.round(2)


nans, x= nan_helper(y)

y[nans]= np.interp(x(nans), x(~nans), y[~nans])

print y.round(2)
'''
'''
for d in range(10):
    weight = 2.8/(d**2+3.0) #0.9, 045, 0.3, 0.25
    #print ''
    #print 'weight',weight
    w = 1-weight
    print weight


xArr = np.array([  0,   np.NaN,  np.NaN,  3. ,  np.NaN ,  5. ,  6. ,  7.  , np.NaN ,  9. , 10. , 11. , np.NaN , 13. , 14. , 15.])
seasonArr = np.array([  12.,   2. ,  3. ,  12.])
seasons = 4
offset =  0
print InterpolateLinearSeasonsNaN(xArr,seasonArr,offset,seasons)



yArr = np.array([  np.NaN,   1. ,  2. ,  3. ,  np.NaN ,  5. ,  6. ,  7.  , np.NaN ,  9. , 10. , 11. , np.NaN , 13. , 14. , 15.])
dstArr = np.array([  0.,   0. ,  0. ,  0.])
seasons = 4
print ResampleSeasonalAvg(yArr, dstArr, seasons)


ts = np.array([ 14,  15.,  16.,  17.,  18.,  19.,  20.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28., 29.])
yArr = np.array([  0.,   1. ,  2. ,  3. ,  4. ,  5. ,  6. ,  7.  , 8. ,  9. , 10. , 11. , 12. , 13. , 14. , 15.])
olsArray = np.array([ -5.00000000e-01 ,  1.55000000e+01 ,  1.47058826e-03 ,  4.60638142e+00])
print OLSextendedNumba(ts,yArr,olsArray)

ts = np.vstack((yArr,ts)).T

b0, b1 = coefficients(ts)
print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))



x = np.arange(120)
y = np.arange(12)
years = 10
step = 12
y = ToAnnualAverage(x, y, step, years)
print y
#print np.allclose(x[:,mycol], y)  # True


x = np.arange(10)
y = np.arange(10)
y = Zscore(x,y)

print y


x = np.arange(10)
x[2] = 0
y = np.arange(10)

olsArr = np.zeros( ( 4 ), np.float32)

b0, b1, r2, rmse = OLSextendedNumba(x,y,olsArr)
print('Coefficients: B0=%.3f, B1=%.3f r2=%.3f RMSE=%.3f ' % (b0, b1, r2, rmse))

ts = np.vstack((x, y)).T

for pair in ts:
    print 'pair', pair
    
for i in range(len( ts)):
    print 'i', ts[i][0]
print ''


#print 'ts',ts[None, :0 ]

  
# calculate coefficients
#dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
CorrCoeffNumba(x,y)

b0, b1,rmse = Coefficients(ts)
print('Coefficients: B0=%.3f, B1=%.3fm RMSE=%.3f ' % (b0, b1,rmse))


b0, b1, r2, rmse = OLSextendedNumba(x,y)
print('Coefficients: B0=%.3f, B1=%.3f r2=%.3f RMSE=%.3f ' % (b0, b1, r2, rmse))


print 'template r2', R2_python_template(x,y)
print 'numpy r2',get_r2_numpy_manual(x,y)

r2 = R2_python(x,y)
print 'r2',r2

#m, c = np.linalg.lstsq(ts, y)[0]
#print m,c
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

print slope, intercept, r_value, p_value, std_err

r = OLSregr(ts)
print r
'''