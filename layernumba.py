'''
Created on 19 Oct 2018

@author: thomasgumbricht
'''
import numba
#import numpy as np

@numba.jit(nopython=True)
def ImageTransform(A, B, offset, scalefac):
    A += (B + offset)*scalefac      
    return A

@numba.jit(nopython=True)
def ImageFgBg(rescalefac, sinrang, cosrang, X, Y, intercept, calibfac):
    A = rescalefac * ( (sinrang*(X+Y-intercept) + cosrang*(-X+Y-intercept)) / 
                             (sinrang*(X-Y+intercept) + cosrang*( X+Y-intercept) + calibfac ) )
    return A


'''@numba.jit(nopython=True)
TODO Not implemented in Numba
'''
def ScalarTWIpercent(B,scalefac, constant, divisor, power, powfac, dstmax):
    A = scalefac * ((B + constant)/divisor + pow(power,(B+constant)*powfac)) 
    A[B < -constant] = 0
    A[A > dstmax] = dstmax
    return A

'''
@numba.jit(nopython=True)
'''
def SingleMask(MASK, BAND, maskNull, bandNull):
    MASK[BAND == bandNull] = maskNull     
    return MASK

'''
@numba.jit(nopython=True)
'''
def AddToMask(MASK, BAND, maskNull, bandNull):   
    MASK[(BAND == bandNull) | (MASK == maskNull)] = maskNull    
    return MASK

'''
@numba.jit(nopython=True)
'''
def SetMask(MASK, BAND, maskNull, bandNull):
    BAND[MASK == maskNull] = bandNull    
    return BAND
