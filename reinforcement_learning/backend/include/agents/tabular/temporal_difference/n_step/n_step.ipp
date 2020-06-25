#pragma once

#include "n_step.hpp"

namespace rl {

namespace agents {

NStep::NStep(size_t states, size_t actions, float alpha, float epsilon, float gamma, int64_t n_step, float epsilon_decay)
    : TemporalDifference(states, actions, alpha, epsilon, gamma, epsilon_decay), n_step(n_step) {}

NStep::NStep(const NStep &other) : TemporalDifference(other), n_step(other.n_step), T(other.T), trajectory(other.trajectory) {}

NStep::~NStep() {}

void NStep::reset(size_t init_state) {
    state = init_state;
    action = select_action(pi, state);

    T = std::numeric_limits<int64_t>::max();
    
    trajectory.clear();
    trajectory.push_back({state, action, 0});
}

std::string NStep::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << demangled << "(alpha=" << alpha << ", epsilon=" << epsilon;
    out << ", gamma=" << gamma << ", n_step=" << n_step << ", epsilon_decay=" << epsilon_decay << ")>";
    return out.str();
}

}  // namespace agents

}  // namespace rl