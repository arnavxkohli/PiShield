from typing import List, Tuple, Dict, Set
from copy import deepcopy
from pishield.linear_requirements.classes import Variable, Atom, Inequality, Constraint

ALLOWED_BOOL_OPS = ['or', 'neg']
ALLOWED_OPS = ['+', '-', '*', '/']
ALLOWED_INEQ_SIGNS = ['>=', '>']


def neg_postprocess_ineq(ineq: Inequality) -> Inequality:
    # equivalent to negating the sign of the inequality and the multiplying by -1
    # e.g. neg x1+x2-x3 >=0 becomes -x1-x2+x3 > 0
    atoms_list, ineq_sign, constant = ineq.get_ineq_attributes()

    # negate all atoms in the inequality
    neg_atoms_list = []
    for atom in atoms_list:
        variable, coefficient, positive_sign = atom.get_atom_attributes()
        neg_atom = Atom(variable, coefficient, not positive_sign)
        neg_atoms_list.append(neg_atom)

    # change the sign of the inequality
    neg_ineq_sign = '>' if ineq_sign == '>=' else '>='
    neg_ineq = Inequality(neg_atoms_list, neg_ineq_sign, constant)
    return neg_ineq


def parse_atom(x):
    """Example: y1, -2y2, -y23, 6y3."""
    x = x.strip().replace(" ", "")
    if x == '':
        return None
    positive_sign = False if '-' in x else True

    if '-' in x or '+' in x:
        x = x[1:]

    coefficient, var_id = x.split('y')
    coefficient = float(coefficient) if coefficient != '' else 1
    var = Variable('y' + var_id)

    atom = Atom(var, coefficient, positive_sign)
    return atom


def parse_inequality(inequality):
    """Example: y1-2y2>0, y1+y2>=0."""
    ineq_sign = None
    inequality = inequality.strip()
    for sign in ALLOWED_INEQ_SIGNS:
        if sign in inequality:
            ineq_sign = sign
            break
    body, constant = inequality.split(ineq_sign)
    constant = float(constant)

    atoms_list = []
    constructed_atom = ''
    for elem in body:
        if elem in ALLOWED_OPS:
            if constructed_atom == '':
                constructed_atom += elem
            else:
                atom = parse_atom(constructed_atom)
                if atom is not None:
                    atoms_list.append(atom)
                constructed_atom = elem
        else:
            constructed_atom += elem
    atom = parse_atom(constructed_atom)
    if atom is not None:
        atoms_list.append(atom)

    ineq = Inequality(atoms_list, ineq_sign, constant)
    return ineq


def parse_constraint(constr) -> Constraint:
    ineqs = []
    neg_postprocess = False

    disjunct_inequalities = constr.split('or')
    for ineq in disjunct_inequalities:
        if 'neg' in ineq:
            ineq = ineq.split('neg')[-1]
            neg_postprocess = True
        ineq = parse_inequality(ineq)
        if neg_postprocess:
            ineq = neg_postprocess_ineq(ineq)
            neg_postprocess = False
        ineqs.append(ineq)

    constr = Constraint(ineqs)
    return constr


def parse_constraints_file(file: str) -> Tuple[List[Variable], List[Constraint]]:
    f = open(file, 'r')
    constraints = []
    ordering = []
    for line in f:
        line = line.strip()
        if 'ordering' in line:
            str_ordering = line.split('ordering ')[-1].split()
            for elem in str_ordering:
                ordering.append(Variable(elem.replace(" ", ""))) # remove all whitespace from variable string
            continue
        constraints.append(parse_constraint(line))
    if len(ordering) == 0:
        raise Exception('An ordering of the variables must be provided!')
    return ordering, constraints


def remap_constraint_variables(ordering: List[Variable],
                               constraints: List[Constraint]) -> Tuple[List[Variable],
                                                                       List[Constraint],
                                                                       Dict[str, str]]:
    # Assumption: Variables are of the form letter_number. Letter can be any letter, preferably y.
    # Note: The mappings are strings, the ordering is a list of variables.
    if not ordering:
        raise Exception("An ordering of the variables must be provided!")

    # Active variables are those that are actually involved within the constraints
    active_variables: Set[str] = set()
    for inequality in map((lambda c: c.single_inequality), constraints):
        active_variables.update(var.readable() for var in inequality.get_body_variables())

    # Build a mapping from original variable names to new variable names of the form y_i
    variable_mapping: Dict[str, str] = {}
    new_index: int = 0
    for var in map((lambda v: v.readable()), ordering):
        if var in active_variables:
            variable_mapping[var] = f"y_{new_index}"
            new_index += 1

    # Perform the remapping with a deepcopy to ensure there isn't any mutation of the original constraints
    remapped_constraints = deepcopy(constraints)
    for constraint in remapped_constraints:
        for atom in constraint.single_inequality.body:
            original_var = atom.variable.readable()
            if original_var not in variable_mapping:
                raise Exception(f"Unexpected variable '{original_var}' not in mapping.")
            atom.variable = Variable(variable_mapping[original_var])

    # Build a new variable ordering that includes only remapped active variables
    remapped_ordering = [
        Variable(variable_mapping[var.readable()])
        for var in ordering
        if var.readable() in variable_mapping
    ]

    # Construct a reverse mapping from remapped names to original names
    # Used to ensure the tensor shape stays same in the ShiedLayer
    reverse_mapping = {
        mapped_var: original_var
        for original_var, mapped_var in variable_mapping.items()
    }

    return remapped_ordering, remapped_constraints, reverse_mapping


def split_constraints(ordering: List[Variable], constraints: List[Constraint]):
    """
    Splits a list of constraints into groups Gi of lists of constraints,
    such that Vars(Gi) \intersect Vars(Gj) = null.
    """
    constr_vars = {}
    for i,constr in enumerate(constraints):
        vars_in_constr = constr.single_inequality.get_body_variables()
        constr_vars[i] = set([var.id for var in vars_in_constr])

    clustered_constraints_ids = list(range(len(constraints)))
    for constr_i in range(len(constraints)):
        vars_in_constr_i = constr_vars[constr_i]
        for constr_j in range(constr_i+1, len(constraints)):

            vars_in_constr_j = constr_vars[constr_j]
            if vars_in_constr_i.intersection(vars_in_constr_j):
                clustered_constraints_ids[constr_j] = clustered_constraints_ids[constr_i]

    clustered_constraints = {id:[] for id in set(clustered_constraints_ids)}
    clustered_orderings = {id:[] for id in set(clustered_constraints_ids)}
    var_ordering = [var.id for var in ordering]
    for cluster_id in clustered_constraints:
        ids_cluster_components = [id for id, group_id in enumerate(clustered_constraints_ids) if group_id == cluster_id]
        clustered_constraints[cluster_id] = [constraints[cluster_component] for cluster_component in ids_cluster_components]

        cluster_ordering = set.union(*[constr_vars[constr_i] for constr_i in ids_cluster_components])
        clustered_orderings[cluster_id] = [var for var in var_ordering if var in cluster_ordering]

    clustered_constraints = list(clustered_constraints.values())
    clustered_orderings = list(clustered_orderings.values())
    return clustered_orderings, clustered_constraints

