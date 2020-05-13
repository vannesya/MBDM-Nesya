'''
Created on May 2, 2017

@author: jhkwakkel
'''
import math
import numpy as np

from scipy.optimize import brentq

def lake_problem005(b=0.42, q=2.0, mean=0.02, stdev=0.0017, delta=0.98,      
                 alpha=0.4, nsamples=100, steps=100, l0= 0.05, l1= 0.05, l2= 0.05, l3= 0.05,
                 l4= 0.05, l5= 0.05, l6= 0.05, l7= 0.05, l8= 0.05, l9= 0.05, l10= 0.05, l11= 0.05, l12= 0.05, l13= 0.05,
                 l14= 0.05, l15= 0.05, l16= 0.05, l17= 0.05, l18= 0.05, l19= 0.05, l20= 0.05, l21= 0.05, l22= 0.05,
                 l23= 0.05, l24= 0.05, l25= 0.05, l26= 0.05, l27= 0.05, l28= 0.05, l29= 0.05, l30= 0.05, l31= 0.05,
                 l32= 0.05, l33= 0.05, l34= 0.05, l35= 0.05, l36= 0.05, l37= 0.05, l38= 0.05, l39= 0.05, l40= 0.05,
                 l41= 0.05, l42= 0.05, l43= 0.05, l44= 0.05, l45= 0.05, l46= 0.05, l47= 0.05, l48= 0.05, l49= 0.05,
                 l50= 0.05, l51= 0.05, l52= 0.05, l53= 0.05, l54= 0.05, l55= 0.05, l56= 0.05, l57= 0.05, l58= 0.05,
                 l59= 0.05, l60= 0.05, l61= 0.05, l62= 0.05, l63= 0.05, l64= 0.05, l65= 0.05, l66= 0.05, l67= 0.05,
                 l68= 0.05, l69= 0.05, l70= 0.05, l71= 0.05, l72= 0.05, l73= 0.05, l74= 0.05, l75= 0.05, l76= 0.05,
                 l77= 0.05, l78= 0.05, l79= 0.05, l80= 0.05, l81= 0.05, l82= 0.05, l83= 0.05, l84= 0.05, l85= 0.05,
                 l86= 0.05, l87= 0.05, l88= 0.05, l89= 0.05, l90= 0.05, l91= 0.05, l92= 0.05, l93= 0.05, l94= 0.05,
                 l95= 0.05, l96= 0.05, l97= 0.05, l98= 0.05, l99= 0.05):   
    decisions = np.array([l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13,
                          l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25,
                          l26, l27, l28, l29, l30, l31, l32, l33, l34, l35, l36, l37,
                          l38, l39, l40, l41, l42, l43, l44, l45, l46, l47, l48, l49,
                          l50, l51, l52, l53, l54, l55, l56, l57, l58, l59, l60, l61,
                          l62, l63, l64, l65, l66, l67, l68, l69, l70, l71, l72, l73,
                          l74, l75, l76, l77, l78, l79, l80, l81, l82, l83, l84, l85,
                          l86, l87, l88, l89, l90, l91, l92, l93, l94, l95, l96, l97,
                          l98, l99])
    
    Pcrit = brentq(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(decisions)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(decisions)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in range(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] +\
                    natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    inertia = np.sum(np.abs(np.diff(decisions)) > 0.02)/float(nvars-1)

    return max_P, utility, inertia, reliability
