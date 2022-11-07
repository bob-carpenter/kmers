from typing import Dict, Literal, Type, Union, get_args

from multinomial_model import multinomial_model
from multinomial_simplex_model import multinomial_simplex_model


class Logistic(multinomial_model):
    def logp_grad(self, theta=None, nograd=False):
        f, g = super().logp_grad(theta)
        if nograd:
            return f
        else:
            return f, g


class Simplex(multinomial_simplex_model):
    pass


KmerModel = Union[Logistic, Simplex]
KmerModels = get_args(KmerModel)
KmerModelName = Literal["Logistic", "Simplex"]
model_classes: Dict[KmerModelName, Type[KmerModel]] = {
    "Logistic": Logistic,
    "Simplex": Simplex,
}
