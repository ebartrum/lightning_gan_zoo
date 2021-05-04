import torch
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

    def transform(self, x, deformed_verts, mean_shape_verts,
            deformation_parameters):
        deformed_verts = deformed_verts[:,::self.template_subdivision]
        return tps_functions.transform(x, deformed_verts,
                deformation_parameters)

class KernelDeformer(nn.Module):
    def __init__(self, template_subdivision, sigma, normalised=False):
        super(KernelDeformer, self).__init__()
        self.template_subdivision = template_subdivision
        self.sigma = sigma
        self.normalised = normalised
        
    def kernel(self, x, y):
        exponent = -torch.abs(x-y) / self.sigma**2
        return torch.exp(exponent)

    def calculate_deformation(self, shape_analysis):
        return None

    def kernel_based_transform(self, x, deformed_verts, mean_shape_verts):
        kernel_res = self.kernel(x.unsqueeze(1), deformed_verts.unsqueeze(2))
        out = (mean_shape_verts.unsqueeze(2)*kernel_res).sum(1)
        if self.normalised:
            denominator = kernel_res.sum(1)
            out = out / denominator
        return out

    def transform(self, x, deformed_verts, mean_shape_verts,
            deformation_parameters):
        deformed_verts = deformed_verts[:,::self.template_subdivision]
        mean_shape_verts = mean_shape_verts[:,::self.template_subdivision]
        return self.kernel_based_transform(x, deformed_verts,
                mean_shape_verts)

class RBFDeformer(nn.Module):
    def __init__(self, template_subdivision, lambda_):
        super(KernelDeformer, self).__init__()
        self.template_subdivision = template_subdivision
        self.lambda_ = lambda_
        
    def calculate_deformation(self, shape_analysis):
        verts = shape_analysis['verts'][:,::self.template_subdivision]
        template_verts =\
            shape_analysis['mean_shape'][:,::self.template_subdivision]
        coefficient = tps_functions.find_coefficients(
            verts, template_verts, self.lambda_).detach()
        return coefficient

    def transform(self, x, deformed_verts, mean_shape_verts,
            deformation_parameters):
        import ipdb;ipdb.set_trace()
        deformed_verts = deformed_verts[:,::self.template_subdivision]
        return tps_functions.transform(x, deformed_verts,
                deformation_parameters)
