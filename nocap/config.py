

class DictToClass:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)


default_config_ = {
    'seed': 20231206,
    'n_head': 4,
    'n_layer': 2,
    'n_ctx': 630, # maximum context length
    'n_positions': 630,
    'dim_embed': 32 * 4,
    'layer_norm_epsilon': 0.00001
}
default_config = DictToClass(default_config_)
