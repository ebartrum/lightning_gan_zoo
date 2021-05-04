from torch import nn
from core.submodules.tps_deformation.tps import functions as tps_functions

class TPSDeformer(nn.Module):
    def __init__(self, template_subdivision, lambda_):
        super(TPSDeformer, self).__init__()
        self.template_subdivision = template_subdivision
        self.lambda_ = lambda_
        
    def calculate_deformation(self, shape_analysis):
        verts = shape_analysis['verts'][:,::self.template_subdivision]
        template_verts =\
            shape_analysis['mean_shape'][:,::self.template_subdivision]
        coefficient = tps_functions.find_coefficients(
            verts, template_verts, self.lambda_).detach()
        return coefficient

    def transform(self, x, deformed_verts, deformation_parameters):
        deformed_verts = deformed_verts[:,::self.template_subdivision]
        return tps_functions.transform(x, deformed_verts,
                deformation_parameters)
