#pragma once

#include "../temporal_difference.ipp"

namespace rl {

namespace agents {

class NStep : public TemporalDifference {
   protected:
    int64_t n_step;
    int64_t T;

    Trajectory trajectory;

   public:
    NStep(size_t states, size_t actions, float alpha, float epsilon, float gamma, int64_t n_step, float epsilon_decay);
    NStep(const NStep &other);
    virtual ~NStep();

    void reset(size_t init_state);

    std::string __repr__();
};

}  // namespace agents

}  // namespace rl