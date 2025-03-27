import random
import torch
from pishield.linear_requirements.classes import Variable, Atom, Inequality, Constraint


def generate_sample(variables: dict[str, Variable], constraints: Constraint) -> dict[str, float] | None:
    sample = {}

    for var_name in variables:
        sample[var_name] = random.uniform(-2, 2)

    max_id = max(variable.id for variable in variables.values())
    tensor_sample = torch.zeros((1, max_id + 1))

    for var_name, value in sample.items():
        tensor_sample[0, variables[var_name].id] = value

    if all(inequality.check_satisfaction(tensor_sample) for inequality in constraints.inequality_list):
        return sample


def monte_carlo_simulation(variables: dict[str, Variable], constraints: Constraint, loss, num_sample: int = 100000) -> float:
    positives, negatives, zeros = 0, 0, 0
    valid_samples = 0

    for _ in range(num_sample):
        sample = generate_sample(variables, constraints)
        if sample is not None:
            valid_samples += 1
            # loss_value = loss(sample)
            loss_value = 0
            positives += loss_value > 0
            negatives += loss_value < 0
            zeros += loss_value == 0

    if valid_samples == 0:
        print("No valid samples found!")
        return 0.0

    print(f"Valid samples: {valid_samples}/{num_sample}")
    print(f"Positives: {positives * 100 / valid_samples:.2f}%")
    print(f"Negatives: {negatives * 100 / valid_samples:.2f}%")
    print(f"Zeros: {zeros * 100 / valid_samples:.2f}%")

    return valid_samples / num_sample


variables = {
    "y_1": Variable("y_1"),
    "y_2": Variable("y_2"),
    "y_3": Variable("y_3"),
    "h_1": Variable("h_1"),
    "h_2": Variable("h_2"),
    "h_3": Variable("h_3"),
}

constraints = Constraint([
        Inequality([Atom(variables["y_1"], 1, True), Atom(variables["h_1"], 1, False)], '>', 0),
        Inequality([Atom(variables["y_2"], 1, True), Atom(variables["h_2"], 1, False)], '>', 0),
        Inequality([Atom(variables["y_3"], 1, True), Atom(variables["h_3"], 1, False)], '<', 0),
        Inequality([Atom(variables["y_1"], 1, True), Atom(variables["y_2"], 1, False), Atom(variables["y_3"], 1, False)], '>', 0),
        Inequality([Atom(variables["h_2"], 1, True), Atom(variables["h_3"], 1, True), Atom(variables["h_1"], 1, False)], '>', 0),
    ]
)

def loss_function(variables: dict[str, float]) -> float:
    return variables["y_1"] - variables["y_2"] + variables["y_3"] - variables["h_2"] + variables["h_3"] - variables["h_1"]

if __name__ == "__main__":
    monte_carlo_simulation(variables, constraints, loss_function)
    print("\nExample constraints:")
    for i, constraint in enumerate(constraints.inequality_list):
        print(f"{i}: {constraint.readable()}")
