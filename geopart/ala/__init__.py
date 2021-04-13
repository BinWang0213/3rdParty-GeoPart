

class StokesHeatModel:

    def __init__(self, eta=None, f_u=None, kappa=None, f_T=None):
        self.eta = eta
        self.f_u = f_u
        self.f_T = f_T
        self.kappa = kappa