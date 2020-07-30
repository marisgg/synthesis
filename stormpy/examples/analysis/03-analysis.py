import stormpy
import stormpy.core

import stormpy.examples
import stormpy.examples.files


def example_analysis_03():
    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)

    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)
    model = stormpy.build_model(prism_program, properties)
    env = stormpy.Environment()
    env.solver_environment.set_linear_equation_solver_type(stormpy.EquationSolverType.native)
    env.solver_environment.native_solver_environment.method = stormpy.NativeLinearEquationSolverMethod.power_iteration
    env.solver_environment.native_solver_environment.maximum_iterations = 2
    result = stormpy.model_checking(model, properties[0], environment=env)
    print(result.at(model.initial_states[0]))

    dd_model = stormpy.build_symbolic_model(prism_program, properties)
    result = stormpy.model_checking(dd_model, properties[0], environment=env)
    filter = stormpy.create_filter_initial_states_symbolic(dd_model)
    result.filter(filter)
    assert result.min == result.max
    print(result.min)



if __name__ == '__main__':
    example_analysis_03()