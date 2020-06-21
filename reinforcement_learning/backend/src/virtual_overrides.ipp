#pragma once

#include <pybind11/pybind11.h>

#include <backend.hpp>

namespace rl {

namespace agents {

class PyAgent : public Agent {
   public:
    using Agent::Agent;
    
    size_t get_action() override { PYBIND11_OVERLOAD(size_t, Agent, get_action, ); }
    void reset(size_t init_state) override { PYBIND11_OVERLOAD_PURE(void, Agent, reset, init_state); }
    void update(size_t next_state, float reward, bool done, bool training) override {
        PYBIND11_OVERLOAD_PURE(void, Agent, update, next_state, reward, done, training);
    }
    std::string __repr__() override {
        PYBIND11_OVERLOAD(std::string, Agent, __repr__, );
    }
};

class PyMonteCarlo : public MonteCarlo {
   public:
    using MonteCarlo::MonteCarlo;

    void control() override { PYBIND11_OVERLOAD_PURE(void, MonteCarlo, control, ); }
};

}  // namespace agents

}  // namespace rl