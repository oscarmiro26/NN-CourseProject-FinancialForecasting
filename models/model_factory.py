from models.mlp_model import get_mlp_model
from models.gru_model import get_gru_model

class ModelFactory:
    @staticmethod
    def create_model(model_str, *args):
        if model_str == 'MLP':
            return get_mlp_model(*args)
        elif model_str == 'GRU':
            return get_gru_model(*args)
        else:
            raise ValueError(f"Model '{model_str}' not found")
