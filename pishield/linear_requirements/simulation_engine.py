import torch
import random
from pishield.linear_requirements.classes import Variable, Atom, Inequality, Constraint


class Prediction:
    def __init__(self, variable: Variable, dependency: Variable, is_greater: bool = True):
        self.variable = variable
        self.dependency = dependency
        self.is_greater = is_greater

    def get_inequality(self) -> Inequality:
        return Inequality([Atom(self.dependency, 1, True), Atom(self.variable, 1, False)], '>' if self.is_greater else '<', 0)


def generate_samples(predictions: dict[str, Prediction], ground_truth: dict[str, Variable], constraints: Constraint) -> dict[str, float] | None:
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have the same length")

    samples = {}
    for gt_variable in ground_truth:
        samples[gt_variable] = random.uniform(-10, 10)

    for prediction_variable, prediction in predictions.items():
        samples[prediction_variable] = samples[prediction.dependency.variable] + (1 if prediction.is_greater else -1) * random.uniform(0.1, 5)

    if constraints.check_satisfaction(samples):
        return samples


def monte_carlo_simulation(predictions: dict[str, Prediction], ground_truth: dict[str, Variable], constraints: Constraint, loss, num_samples: int = 1000) -> float:
    positives, negatives, zeros = 0, 0, 0
    for _ in range(num_samples):
        samples = generate_samples(predictions, ground_truth, constraints)
        if samples is not None:
            loss_value = loss(samples) # TODO: add loss function substitution
            positives += loss_value > 0
            negatives += loss_value < 0
            zeros += loss_value == 0

    valid_samples = positives + negatives + zeros
    print(f"Positives: {positives * 100 / valid_samples}%")
    print(f"Negatives: {negatives * 100 / valid_samples}%")
    print(f"Zeros: {zeros * 100 / valid_samples}%")


predictions = {
    "h_1": Prediction(Variable("h_1"), Variable("y_1"), False),
    "h_2": Prediction(Variable("h_2"), Variable("y_2"), False),
    "h_3": Prediction(Variable("h_3"), Variable("y_3"), True)
}

ground_truth = {
    "y_1": Variable("y_1"),
    "y_2": Variable("y_2"),
    "y_3": Variable("y_3")
}

constraints = Constraint([prediction.get_inequality() for prediction in predictions.values()] + [
        Inequality([Atom(ground_truth["y_1"], 1, True), Atom(ground_truth["y_2"], 1, False), Atom(ground_truth["y_3"], 1, False)], '>', 0),
        Inequality([Atom(predictions["h_2"].variable, 1, True), Atom(predictions["h_3"].variable, 1, True), Atom(predictions["h_1"].variable, 1, False)], '>', 0),
    ]
)
