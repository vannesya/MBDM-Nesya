from ema_workbench import Model, RealParameter, TimeSeriesOutcome, ArrayOutcome, ScalarOutcome

def definemodel(modelname, modelfunction):
    model = Model(modelname, function=modelfunction)
    model.uncertainties = [RealParameter('b', 0.1, 0.45), RealParameter('mean', 0.01, 0.05),
                           RealParameter('stdev', 0.001, 0.005), RealParameter('delta', 0.93, 0.99),
                           RealParameter('q', 2, 4.5)]
    model.levers =  [RealParameter('l0', 0.0, 0.1),RealParameter('l36', 0.0, 0.1),RealParameter('l72', 0.0, 0.1),
                     RealParameter('l1', 0.0, 0.1),RealParameter('l37', 0.0, 0.1),RealParameter('l73', 0.0, 0.1),
                     RealParameter('l2', 0.0, 0.1),RealParameter('l38', 0.0, 0.1),RealParameter('l74', 0.0, 0.1),
                     RealParameter('l3', 0.0, 0.1),RealParameter('l39', 0.0, 0.1),RealParameter('l75', 0.0, 0.1),
                     RealParameter('l4', 0.0, 0.1),RealParameter('l40', 0.0, 0.1),RealParameter('l76', 0.0, 0.1),
                     RealParameter('l5', 0.0, 0.1),RealParameter('l41', 0.0, 0.1),RealParameter('l77', 0.0, 0.1),
                       RealParameter('l6', 0.0, 0.1),RealParameter('l42', 0.0, 0.1),RealParameter('l78', 0.0, 0.1),
                       RealParameter('l7', 0.0, 0.1),RealParameter('l43', 0.0, 0.1),RealParameter('l79', 0.0, 0.1),
                       RealParameter('l8', 0.0, 0.1),RealParameter('l44', 0.0, 0.1),RealParameter('l80', 0.0, 0.1),
                       RealParameter('l9', 0.0, 0.1),RealParameter('l45', 0.0, 0.1),RealParameter('l81', 0.0, 0.1),
                       RealParameter('l10', 0.0, 0.1),RealParameter('l46', 0.0, 0.1),RealParameter('l82', 0.0, 0.1),
                       RealParameter('l11', 0.0, 0.1),RealParameter('l47', 0.0, 0.1),RealParameter('l83', 0.0, 0.1),
                       RealParameter('l12', 0.0, 0.1),RealParameter('l48', 0.0, 0.1),RealParameter('l84', 0.0, 0.1),
                       RealParameter('l13', 0.0, 0.1),RealParameter('l49', 0.0, 0.1),RealParameter('l85', 0.0, 0.1),
                       RealParameter('l14', 0.0, 0.1),RealParameter('l50', 0.0, 0.1),RealParameter('l86', 0.0, 0.1),
                       RealParameter('l15', 0.0, 0.1),RealParameter('l51', 0.0, 0.1),RealParameter('l87', 0.0, 0.1),
                       RealParameter('l16', 0.0, 0.1),RealParameter('l52', 0.0, 0.1),RealParameter('l88', 0.0, 0.1),
                       RealParameter('l17', 0.0, 0.1),RealParameter('l53', 0.0, 0.1),RealParameter('l89', 0.0, 0.1),
                       RealParameter('l18', 0.0, 0.1),RealParameter('l54', 0.0, 0.1),RealParameter('l90', 0.0, 0.1),
                       RealParameter('l19', 0.0, 0.1),RealParameter('l55', 0.0, 0.1),RealParameter('l91', 0.0, 0.1),
                       RealParameter('l20', 0.0, 0.1),RealParameter('l56', 0.0, 0.1),RealParameter('l92', 0.0, 0.1),
                       RealParameter('l21', 0.0, 0.1),RealParameter('l57', 0.0, 0.1),RealParameter('l93', 0.0, 0.1),
                       RealParameter('l22', 0.0, 0.1),RealParameter('l58', 0.0, 0.1),RealParameter('l94', 0.0, 0.1),
                       RealParameter('l23', 0.0, 0.1),RealParameter('l59', 0.0, 0.1),RealParameter('l95', 0.0, 0.1),
                       RealParameter('l24', 0.0, 0.1),RealParameter('l60', 0.0, 0.1),RealParameter('l96', 0.0, 0.1),
                       RealParameter('l25', 0.0, 0.1),RealParameter('l61', 0.0, 0.1),RealParameter('l97', 0.0, 0.1),
                       RealParameter('l26', 0.0, 0.1),RealParameter('l62', 0.0, 0.1),RealParameter('l98', 0.0, 0.1),
                       RealParameter('l27', 0.0, 0.1),RealParameter('l63', 0.0, 0.1),RealParameter('l99', 0.0, 0.1),
                       RealParameter('l28', 0.0, 0.1),RealParameter('l64', 0.0, 0.1),
                       RealParameter('l29', 0.0, 0.1),RealParameter('l65', 0.0, 0.1),
                       RealParameter('l30', 0.0, 0.1),RealParameter('l66', 0.0, 0.1),
                       RealParameter('l31', 0.0, 0.1),RealParameter('l67', 0.0, 0.1),
                       RealParameter('l32', 0.0, 0.1),RealParameter('l68', 0.0, 0.1),
                       RealParameter('l33', 0.0, 0.1),RealParameter('l69', 0.0, 0.1),
                       RealParameter('l34', 0.0, 0.1),RealParameter('l70', 0.0, 0.1),
                       RealParameter('l35', 0.0, 0.1),RealParameter('l71', 0.0, 0.1)] 

    model.outcomes = [ScalarOutcome('max_P'), ScalarOutcome('utility'), ScalarOutcome('inertia'), ScalarOutcome('reliability')]
    return model