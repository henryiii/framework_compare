// Compile and run with:
//   clang++ -std=c++14 -O3 compile.cpp
//   ./a.out

#include <random>
#include <array>
#include <chrono>
#include <tuple>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 10000;
const int nEpochs = 10000;

tuple<float, float> gradientDescent(array<float, N> x, array<float, N> d, float mu) {
    array<float, N> y{};
    const float f = 2.f / N;

    float err;
    float w0 = 0.f;
    float w1 = 0.f;
    float grad0;
    float grad1;

    for(int n=0; n<nEpochs; n++) {
        grad0 = 0.f;
        grad1 = 0.f;

        for(int i=0; i<N; i++) {
            err = f * (d[i] - y[i]);
            grad0 += err;
            grad1 += err * x[i];
        }

        w0 += mu * grad0;
        w1 += mu * grad1;

        for(int i=0; i<N; i++) {
            y[i] = w0 + w1 * x[i];
        }
    }

    return {w0, w1};
}

int main() {
    const float f = 2.f / N;

    const float sigma = 0.1f;
    const float mu = 0.001f;

    // Generate
    default_random_engine e1(444);
    uniform_real_distribution<float> uni_dist;

    array<float, N> x;
    for(int i=0; i<N; i++)
        x[i] = i * f;
    
    array<float, N> d;
    transform(x.begin(), x.end(), d.begin(),
              [&](float v){return 3.f + 2.f*v + sigma*uni_dist(e1);});

    cout << "Running C++ example" << endl;

    // Start timer
    auto start = std::chrono::system_clock::now();
    auto result = gradientDescent(x, d, mu);
    auto stop = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = stop - start;
    cout << "  Solve time: " << elapsed_seconds.count() << endl;
    
    cout << "  Answer: w_0=" << get<0>(result) << ", w_1=" << get<1>(result) << endl;
    return 0;
}
