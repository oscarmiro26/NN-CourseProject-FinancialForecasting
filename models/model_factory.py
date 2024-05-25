from models.mlp_model import get_mlp_model
from .gru_model import get_gru_model

class ModelFactory:
    @staticmethod
    def create_model(
        model_str,
        *args
    ):
        model = None
        if model_str == 'GRU':
            model = get_gru_model(*args)
        elif model_str == 'MLP':
            model = get_mlp_model(*args)
        else:
            raise ValueError(f'Model \'{model_str}\' not found')

        return model
        