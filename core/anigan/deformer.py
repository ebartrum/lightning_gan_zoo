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

def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


def init_out_weights(self):
    for m in self.modules():
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -1e-5, 1e-5)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

class LSTMDeformer(nn.Module):
    def __init__(self, template_subdivision):
        super(LSTMDeformer, self).__init__()
        self.template_subdivision = template_subdivision
        self.latent_size = 128
        self.warper = Warper(latent_size=128, hidden_size=128, steps=4)
        self.offset_mlp = nn.Sequential(
          nn.Linear(243, 512),
          nn.ReLU(),
          nn.Linear(512, 128))
        
    def calculate_deformation(self, shape_analysis):
        verts = shape_analysis['verts'][:,::self.template_subdivision]
        template_verts =\
            shape_analysis['mean_shape'][:,::self.template_subdivision]
        offsets = verts.flatten(1) - template_verts.flatten(1)
        latent = self.offset_mlp(offsets) 
        return latent

    def transform(self, x, deformed_verts, mean_shape_verts,
            deformation_parameters):
        n_ray_pts = x.shape[1]
        deformation_parameters.unsqueeze(1).repeat(1,n_ray_pts,1)
        combined_deformation_x = torch.cat((deformation_parameters.unsqueeze(1).repeat(1,n_ray_pts,1),x),-1)
        warper_input = combined_deformation_x.flatten(end_dim=1)
        n_chunks = len(warper_input)//6000
        warper_input_chunks = torch.chunk(warper_input, n_chunks, dim=0)
        warp_chunks = [self.warper(chk)[-1][-1] for chk in warper_input_chunks]
        out = torch.cat(warp_chunks)
        out = out.reshape(x.shape)
        return out

class Warper(nn.Module):
    def __init__(
            self,
            latent_size,
            hidden_size,
            steps,
    ):
        super(Warper, self).__init__()
        self.n_feature_channels = latent_size + 3
        self.steps = steps
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size=self.n_feature_channels,
                                hidden_size=hidden_size)
        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.out_layer_coord_affine = nn.Linear(hidden_size, 6)
        self.out_layer_coord_affine.apply(init_out_weights)

    def forward(self, input, step=1.0):
        if step < 1.0:
            input_bk = input.clone().detach()

        xyz = input[:, -3:]
        code = input[:, :-3]
        states = [None]
        warping_param = []

        warped_xyzs = []
        for s in range(self.steps):
            state = self.lstm(torch.cat([code, xyz], dim=1), states[-1])
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))
            a = self.out_layer_coord_affine(state[0])
            tmp_xyz = torch.addcmul(a[:, 3:], (1 + a[:, :3]), xyz)

            warping_param.append(a)
            states.append(state)
            if (s+1) % (self.steps // 4) == 0:
                warped_xyzs.append(tmp_xyz)
            xyz = tmp_xyz

        if step < 1.0:
            xyz_ = input_bk[:, -3:]
            xyz = xyz * step + xyz_ * (1 - step)

        return xyz, warping_param, warped_xyzs
