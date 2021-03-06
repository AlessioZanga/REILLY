#pragma once

#include "semigradient_sarsa.hpp"

namespace reilly {

namespace agents {

SemiGradientSarsa::SemiGradientSarsa(size_t actions, float alpha, float epsilon, float gamma, float epsilon_decay, py::kwargs kwargs)
    : ApproximateTemporalDifference(actions, alpha, epsilon, gamma, epsilon_decay, kwargs) {}

SemiGradientSarsa::SemiGradientSarsa(const SemiGradientSarsa &other) : ApproximateTemporalDifference(other) {}

SemiGradientSarsa &SemiGradientSarsa::operator=(const SemiGradientSarsa &other) {
    if (this != &other) {
        SemiGradientSarsa tmp(other);
        std::swap(tmp.action, action);
        std::swap(tmp.alpha, alpha);
        std::swap(tmp.epsilon, epsilon);
        std::swap(tmp.gamma, gamma);
        std::swap(tmp.epsilon_decay, epsilon_decay);
        std::swap(tmp.estimator, estimator);
        std::swap(tmp.state, state);
        std::swap(tmp.action, action);
    }
    return *this;
}

SemiGradientSarsa::~SemiGradientSarsa() {}

void SemiGradientSarsa::update(Vector next_state, float reward, bool done, py::kwargs kwargs) {
    size_t next_action = select_action(estimator, next_state);
    
    bool training = py::cast<bool>(kwargs["training"]);
    if (training) {
        float G = reward + gamma * estimator(next_state, next_action);
        estimator.update(state, action, G);
    }

    state = next_state;
    action = next_action;

    if (done) epsilon *= epsilon_decay;
}

}  // namespace agents

}  // namespace reilly