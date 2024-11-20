#include "../synthesis.h"

#include "ObservationEvaluator.h"
#include "FscUnfolder.h"
#include "GameAbstractionSolver.h"

#include "SparseDerivativeInstantiationModelCheckerFamily.h"
#include "GradientDescentInstantiationSearcherFamily.h"
#include <storm/environment/Environment.h>
#include <storm/modelchecker/results/ExplicitQuantitativeCheckResult.h>
#include <storm-pars/utility/FeasibilitySynthesisTask.h>
#include <storm-pars/derivative/GradientDescentMethod.h>

#include <pybind11/functional.h>
#include <future>
#include <thread>

void bindings_pomdp_family(py::module& m) {

    py::class_<synthesis::ObservationEvaluator<double>>(m, "ObservationEvaluator")
        .def(py::init<storm::prism::Program &,storm::models::sparse::Model<double> const& >(), py::arg("prism"), py::arg("model"))
        .def_property_readonly("num_obs_expressions", [](synthesis::ObservationEvaluator<double>& e) {return e.num_obs_expressions;} )
        .def_property_readonly("obs_expr_label", [](synthesis::ObservationEvaluator<double>& e) {return e.obs_expr_label;} )
        .def_property_readonly("obs_expr_is_boolean", [](synthesis::ObservationEvaluator<double>& e) {return e.obs_expr_is_boolean;} )
        .def_property_readonly("num_obs_classes", [](synthesis::ObservationEvaluator<double>& e) {return e.num_obs_classes;} )
        .def_property_readonly("state_to_obs_class", [](synthesis::ObservationEvaluator<double>& e) {return e.state_to_obs_class;} )
        .def("obs_class_value", &synthesis::ObservationEvaluator<double>::observationClassValue, py::arg("obs_class"), py::arg("obs_expr"))
        .def("add_observations_to_submdp", &synthesis::ObservationEvaluator<double>::addObservationsToSubMdp, py::arg("mdp"), py::arg("state_sub_to_full"))
        ;

    py::class_<synthesis::FscUnfolder<double>>(m, "FscUnfolder")
        .def(
            py::init<storm::models::sparse::Model<double> const&,
            std::vector<uint32_t> const&,
            uint64_t,
            std::vector<uint64_t> const&>()
        )
        .def("apply_fsc", &synthesis::FscUnfolder<double>::applyFsc, py::arg("action_function"), py::arg("udate_function"))
        .def_property_readonly("product", [](synthesis::FscUnfolder<double>& m) {return m.product;} )
        .def_property_readonly("product_choice_to_choice", [](synthesis::FscUnfolder<double>& m) {return m.product_choice_to_choice;} )
        // .def_property_readonly("product_state_to_state", [](synthesis::FscUnfolder<double>& m) {return m.product_state_to_state;} )
        // .def_property_readonly("product_state_to_state_memory_action", [](synthesis::FscUnfolder<double>& m) {return m.product_state_to_state_memory_action;} )
        ;

    // m.def("randomize_action_variant", &synthesis::randomizeActionVariant<double>);
    py::class_<synthesis::GameAbstractionSolver<double>>(m, "GameAbstractionSolver")
        .def(
            py::init<storm::models::sparse::Model<double> const&, uint64_t, std::vector<uint64_t> const&, std::string const&, double>(),
            py::arg("quotient"), py::arg("quoitent_num_actions"), py::arg("choice_to_action"), py::arg("target_label"), py::arg("precision")
        )
        .def("solve", &synthesis::GameAbstractionSolver<double>::solve)
        .def_property_readonly("solution_state_values", [](synthesis::GameAbstractionSolver<double>& solver) {return solver.solution_state_values;})
        .def_property_readonly("solution_value", [](synthesis::GameAbstractionSolver<double>& solver) {return solver.solution_value;})
        .def_property_readonly("solution_state_to_player1_action", [](synthesis::GameAbstractionSolver<double>& solver) {return solver.solution_state_to_player1_action;})
        .def_property_readonly("solution_state_to_quotient_choice", [](synthesis::GameAbstractionSolver<double>& solver) {return solver.solution_state_to_quotient_choice;})
        .def("enable_profiling", &synthesis::GameAbstractionSolver<double>::enableProfiling)
        .def("print_profiling", &synthesis::GameAbstractionSolver<double>::printProfiling)
        ;


    py::class_<storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction, double>>(m, "SparseDerivativeInstantiationModelCheckerFamily", "Derivatives and stuff.")
        .def(py::init<storm::models::sparse::Dtmc<storm::RationalFunction> const&>(), "Constructor.")
        // .def("check", py::overload_cast<storm::Environment const&, storm::utility::parametric::Valuation<storm::RationalFunction> const&, typename storm::utility::parametric::VariableType<storm::RationalFunction>::type const&>(&storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction, double>::check)
        .def("specifyFormula", &storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction, double>::specifyFormula)
        // .def("check", &storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction, double>::check, py::arg("env"), py::arg("valuation"), py::arg("parameter"), py::arg("valueVector")=boost::none)
        .def("check", [](
            storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction,double> & der,
            storm::Environment const& env,
            storm::utility::parametric::Valuation<storm::RationalFunction> const& valuation,
            typename storm::utility::parametric::VariableType<storm::RationalFunction>::type const& parameter
        ) { return der.check(env,valuation,parameter); })
        .def("checkWithValue", [](
            storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction,double> & der,
            storm::Environment const& env,
            storm::utility::parametric::Valuation<storm::RationalFunction> const& valuation,
            typename storm::utility::parametric::VariableType<storm::RationalFunction>::type const& parameter,
            std::vector<double> const& valueVector
        ) { return der.check(env,valuation,parameter,valueVector); })
        .def("checkMultipleParameters", [](
            storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction,double> & der,
            storm::Environment const& env,
            storm::utility::parametric::Valuation<storm::RationalFunction> const& valuation,
            std::vector<typename storm::utility::parametric::VariableType<storm::RationalFunction>::type> const& parameters,
            std::vector<double> const& valueVector
        ) { 
            std::map<typename storm::utility::parametric::VariableType<storm::RationalFunction>::type, double> gradients = std::map<typename storm::utility::parametric::VariableType<storm::RationalFunction>::type, double>();
            for(typename storm::utility::parametric::VariableType<storm::RationalFunction>::type p : parameters) {
                double grad = (*der.check(env, valuation, p, valueVector)).getValueVector().at(0);
                gradients[p] = grad;
            }

            return gradients;
            })
        .def("checkMultipleParametersMT", [](
            storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction,double> & der,
            storm::Environment const& env,
            storm::utility::parametric::Valuation<storm::RationalFunction> const& valuation,
            std::vector<typename storm::utility::parametric::VariableType<storm::RationalFunction>::type> const& parameters,
            std::vector<double> const& valueVector
        ) { 
            std::map<typename storm::utility::parametric::VariableType<storm::RationalFunction>::type, double> gradients = std::map<typename storm::utility::parametric::VariableType<storm::RationalFunction>::type, double>();
            // std::map<typename storm::utility::parametric::VariableType<storm::RationalFunction>::type, std::future<double>> gradients = std::map<typename storm::utility::parametric::VariableType<storm::RationalFunction>::type, std::future<double>>();

            for(auto p : parameters) {
                // double grad = (*der.check(env, valuation, p, valueVector)).getValueVector().at(0);
                auto myLambda = [](storm::derivative::SparseDerivativeInstantiationModelCheckerFamily<storm::RationalFunction,double> & derr, storm::Environment const& envv, storm::utility::parametric::Valuation<storm::RationalFunction> const& valuationn,  typename storm::utility::parametric::VariableType<storm::RationalFunction>::type const& parameterr, std::vector<double> const& valueVectorr) {
                    return (*derr.check(envv, valuationn, parameterr, valueVectorr)).getValueVector().at(0);
                };
                // std::thread t(std::move(myLambda), std::move(der), std::move(env), std::move(valuation), std::move(p), std::move(valueVector));
                // t.join();
                // auto grad = myLambda(der, env, valuation, p, valueVector);
                // std::future<double> grad = std::async(std::launch::async, &myLambda, der, env, valuation, p, valueVector);
                // auto grad_result = std::async(std::launch::async, myLambda, der, env, valuation, p, valueVector);
                // gradients[p] = grad_result.get().getValueVector().at(0);
                gradients[p] = myLambda(der, env, valuation, p, valueVector);
            }

            return gradients;
            })
        ;

    // std::future<>
    
    // py::class_<storm::derivative::GradientDescentMethod>(m, "GradientDescentMethod"); // TODO
    
    py::class_<storm::pars::FeasibilitySynthesisTask, std::shared_ptr<storm::pars::FeasibilitySynthesisTask>>(m, "FeasibilitySynthesisTask")
        .def(py::init<std::shared_ptr<storm::logic::Formula const> const&>())
        .def("set_bound", [](storm::pars::FeasibilitySynthesisTask& task, storm::logic::ComparisonType comparisonType, storm::expressions::Expression const& bound) {
            task.setBound(storm::logic::Bound(comparisonType, bound));
        })
        ;

    py::class_<storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>>(m, "GradientDescentInstantiationSearcherFamily", "Derivatives wrapper and stuff.")
        .def(py::init<storm::models::sparse::Dtmc<storm::RationalFunction> const&, 
        // storm::derivative::GradientDescentMethod, // TODO
        double, uint_fast64_t, uint_fast64_t>(), "Constructor.")
        // .def(py::init([](
        //     storm::models::sparse::Dtmc<storm::RationalFunction> const& model, double learningRate,
        //     double averageDecay, double squaredAverageDecay, uint_fast64_t miniBatchSize, double terminationEpsilon,
        //     boost::optional<std::map<typename utility::parametric::VariableType<storm::RationalFunction>::type, typename utility::parametric::CoefficientType<storm::RationalFunction>::type>> startPoint, bool recordRun
        // ) {
        //     GradientDescentMethod method = GradientDescentMethod::MOMENTUM_SIGN;
        //     GradientDescentConstraintMethod constraintMethod = GradientDescentConstraintMethod::PROJECT_WITH_GRADIENT;
        //     return GradientDescentInstantiationSearcherFamily(model,method,learningRate,averageDecay,squaredAverageDecay,miniBatchSize,terminationEpsilon,startPoint,constraintMethod,recordRun);
        // }))

        // .def_property("derivativeEvaluationHelper", [](storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>& solver) {return solver.derivativeEvaluationHelper;})
        // .def_property("assignments", [](storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>& solver) {return solver.assignments;})
        // .def_property("assignments", &storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>::assignments)

        .def("setup", &storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>::setup)
        .def("gradientDescent", &storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>::gradientDescent)
        .def("stochasticGradientDescent", &storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>::stochasticGradientDescent)
        // .def("stochasticGradientDescent", [] (
            // storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>& der,
            // // std::map<storm::RationalFunction, storm::utility::parametric::CoefficientType<storm::RationalFunction>>& position
            // std::map<storm::utility::parametric::VariableType<storm::RationalFunction>, storm::utility::parametric::CoefficientType<storm::RationalFunction>>& position
        // ) { return der.stochasticGradientDescent(position);})
        .def("resetDynamicValues", &storm::derivative::GradientDescentInstantiationSearcherFamily<storm::RationalFunction, double>::resetDynamicValues)
        ;

}
