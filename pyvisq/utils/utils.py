import copy
from ..model import Model


def sweep(model: Model, x_to_y: tuple[str, str], values: list[float]) -> dict[float, Model]:
    sweep_dict = dict()
    x, y = x_to_y
    y_value = getattr(model, y)
    for val in values:
        new_model = copy.deepcopy(model)
        x_value = val * y_value
        setattr(new_model, x, x_value)
        new_model.make()
        new_model.run()
        sweep_dict[val] = new_model

    return sweep_dict


map_dict = {
    'A': '\\alpha',
    'B': '\\beta',
    'E1': 'E_1',
    'E2': 'E_2',
    'T1': '\\tau_1',
    'T2': '\\tau_2',
    'EK': 'E_K',
    'CA': 'c_{\\alpha}',
    'CB': 'c_{\\beta}',
    'I': '\\epsilon',
    'D': 't_{ind}',
    'L': 't_{dwell}',
}
