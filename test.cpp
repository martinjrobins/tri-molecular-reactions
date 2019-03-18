#include "Aboria.h"
#include "trase.hpp"
#include <fstream>
#include <random>

int main(void)
{

  const unsigned int D = 2;

  // create the container type to hold all the particles, using a dimension of 2
  // we will also use a std::vector to hold all the variables, and the KdtreeNanoflann
  // spatial data structure
  using particles_t = Aboria::Particles<std::tuple<>, D>;
  using position = particles_t::position;
  using Aboria::alive;
  using Aboria::get;

  const size_t N = 1;
  const int timesteps = 1000;
  const double dt = 0.01;
  const double Diff = 1.0;
  const double diffusion_constant = std::sqrt(2 * Diff * dt);
  const double radiusAB = 0.1;
  const double rho2 = 1.0;
  const double alpha2 = 1.0;
  const double alpha3 = 1.0;
  const double k1 = 1.0;
  particles_t particlesA(N);
  particles_t particlesB(N);
  particles_t particlesC(0);

  // randomly set the particle positions
  const double min = 0;
  const double max = 1;
  std::default_random_engine gen;

  // initialise particles
  auto init = [&](auto& particles) {
    std::uniform_real_distribution<double> uniform(min, max);
    for (int i = 0; i < particles.size(); ++i) {
      get<position>(particles)[i] = Aboria::vdouble2(uniform(gen), uniform(gen));
    }
    // now setup the neighbour search
    auto minv = Aboria::Vector<double, D>::Constant(min);
    auto maxv = Aboria::Vector<double, D>::Constant(max);
    auto periodicv = Aboria::Vector<bool, D>::Constant(true);
    particles.init_neighbour_search(minv, maxv, periodicv, 10);
  };
  init(particlesA);
  init(particlesB);
  init(particlesC);

  // store particle counts
  std::vector<int> nA(timesteps);
  std::vector<int> nB(timesteps);
  std::vector<int> nC(timesteps);

  // simulate
  for (int i = 0; i < timesteps; ++i) {

    // diffuse
    auto diffuse = [&](auto& particles) {
      std::normal_distribution<double> normal(0, 1);
      for (int i = 0; i < particles.size(); ++i) {
        get<position>(particles)[i]
            += diffusion_constant * Aboria::vdouble2(normal(gen), normal(gen));
      }
      particles.update_positions();
    };
    diffuse(particlesA);
    diffuse(particlesB);
    diffuse(particlesC);

    // react
    for (const auto& A : particlesA) {
      for (auto b
           = euclidean_search(particlesB.get_query(), get<position>(A), radiusAB);
           b != false; ++b) {
        const Aboria::vdouble2& dx = b.dx();
        const double r2 = dx.squaredNorm();
        const Aboria::vdouble2 search_point = get<position>(A) + 0.5 * dx;
        const double radiusABC = std::sqrt((rho2 - alpha2 * 0.25 * r2) / alpha3);
        for (auto c = euclidean_search(particlesC.get_query(), search_point, radiusABC);
             c != false; ++c) {
          // react: A + B + C -> A + B
          get<alive>(*c) = false;
        }
      }
    }
    // react: null -> C
    std::poisson_distribution<int> poisson(k1);
    std::uniform_real_distribution<double> uniform(min, max);
    for (int i = 0; i < poisson(gen); ++i) {
      particles_t::value_type newp;
      get<position>(newp) = Aboria::vdouble2(uniform(gen), uniform(gen));
      particlesC.push_back(newp, false);
    }
    particlesC.update_positions();

    // record counts
    nA[i] = particlesA.size();
    nB[i] = particlesB.size();
    nC[i] = particlesC.size();
  }

  // now plot the histograms of counts
  auto plot_hist = [&](const std::vector<int>& counts, const std::string& filename) {
    auto fig = trase::figure();
    auto ax = fig->axis();
    auto data = trase::create_data().x(counts);
    auto hist = ax->histogram(data);
    std::ofstream out;
    out.open(filename.c_str());
    trase::BackendSVG backend(out);
    fig->draw(backend);
    out.close();
  };
  plot_hist(nA, "histA.svg");
  plot_hist(nB, "histB.svg");
  plot_hist(nC, "histC.svg");

  // now plot the time trase of counts
  auto plot_trace = [&](const std::vector<int>& counts, const std::string& filename) {
    auto fig = trase::figure();
    auto ax = fig->axis();
    std::vector<int> iterations(timesteps);
    std::iota(iterations.begin(), iterations.end(), 0);
    auto data = trase::create_data().x(iterations).y(counts);
    auto hist = ax->line(data);
    std::ofstream out;
    out.open(filename.c_str());
    trase::BackendSVG backend(out);
    fig->draw(backend);
    out.close();
  };
  plot_trace(nA, "lineA.svg");
  plot_trace(nB, "lineB.svg");
  plot_trace(nB, "lineC.svg");
}
