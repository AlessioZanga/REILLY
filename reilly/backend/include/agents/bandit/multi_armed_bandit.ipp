#pragma once

#include "multi_armed_bandit.hpp"

namespace reilly {

namespace agents {

template <typename Arm>
MultiArmedBandit<Arm>::MultiArmedBandit(size_t actions, float epsilon_decay) : Agent(actions, 0, 0, 0, epsilon_decay) {}

template <typename Arm>
MultiArmedBandit<Arm>::MultiArmedBandit(const MultiArmedBandit &other) : Agent(other), arms(other.arms) {}

template <typename Arm>
MultiArmedBandit<Arm>::~MultiArmedBandit() {}

template <typename Arm>
void MultiArmedBandit<Arm>::reset(size_t init_state) {
    arms.clear();
    for (size_t i = 0; i < actions; i++) arms.push_back(Arm());
    action = select_action();
}

template <typename Arm>
void MultiArmedBandit<Arm>::update(size_t next_state, float reward, bool done, py::kwargs kwargs) {
    arms[action].update(reward, epsilon_decay);
    for (Arm arm : arms) arm.trace.push_back((float) arm);
    action = select_action();
}

template <typename Arm>
std::string MultiArmedBandit<Arm>::__repr__() {
    int status;
    std::stringstream out;
    char *demangled = abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status);
    out << "<" << arms[0].__repr__() << demangled << "(arms= " << actions;
    out << ", decay=" << epsilon_decay << ")";
    return out.str();
}

}  // namespace agents

}  // namespace reilly