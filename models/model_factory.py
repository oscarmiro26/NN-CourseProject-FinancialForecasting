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

        else:
            raise ValueError('Model not found')

        return model
        