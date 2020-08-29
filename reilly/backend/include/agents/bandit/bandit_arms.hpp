#include "../agent.ipp"

namespace reilly {

namespace agents {

class BanditArm {
   public:
    size_t count;
    std::vector<float> trace;

    BanditArm();
    BanditArm(const BanditArm &other);
    virtual ~BanditArm();

    virtual double operator()(std::minstd_rand &generator) const = 0;
    virtual void update(float reward, float gamma, float decay) = 0;
    virtual float UCB(float T) const = 0;  // Upper Confident Bound
    virtual operator float() const = 0;
};

class BernoulliArm : public BanditArm {
   public:
    float alpha;
    float beta;

    BernoulliArm(float alpha = 1, float beta = 1);
    BernoulliArm(const BernoulliArm &other);
    BernoulliArm &operator=(const BernoulliArm &other);
    ~BernoulliArm();

    virtual double operator()(std::minstd_rand &generator) const;
    virtual void update(float reward, float gamma, float decay);
    virtual float UCB(float T) const;
    virtual operator float() const;
};

class DynamicBernoulliArm : public BernoulliArm {
   public:
    DynamicBernoulliArm(float alpha = 2, float beta = 2);
    DynamicBernoulliArm(const DynamicBernoulliArm &other);
    DynamicBernoulliArm &operator=(const DynamicBernoulliArm &other);
    ~DynamicBernoulliArm();

    virtual void update(float reward, float gamma, float decay);
};

class GaussianArm : public BanditArm {
   private:
    float ri;
    float qi;

   public:
    float mu;
    float stddev;

    GaussianArm(float mu = 0.5, float stddev = 1);
    GaussianArm(const GaussianArm &other);
    GaussianArm &operator=(const GaussianArm &other);
    ~GaussianArm();

    virtual double operator()(std::minstd_rand &generator) const;
    virtual void update(float reward, float gamma, float decay);
    virtual float UCB(float T) const;
    virtual operator float() const;
};

}  // namespace agents

}  // namespace reilly