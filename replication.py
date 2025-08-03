from loglinearcorrection.correction_estimator import DoublyRobustElasticityEstimator

"""
Metadata format:
{
'PaperID' : Numeric,
'Table': Numeric,
'Result Number': Numeric,
'Binary_x': Boolean
}
"""
class Replicator:
    def __init__(self, metadata,  y, X, fe = None, z = None, elasticity=False, replicated=False,