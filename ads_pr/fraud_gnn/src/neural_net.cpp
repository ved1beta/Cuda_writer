#include "neural_net.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

// ============================================================
// neural_net.cpp - MLP implementation with SGD backprop
// ============================================================

// ----------------------------------------------------------
// xavier (helper)
//
// Short:
//   Draw a single Xavier-uniform initial weight.
//
// Detailed:
//   Returns a sample from Uniform[-limit, +limit] where
//   limit = sqrt(6 / (fan_in + fan_out)). This range keeps
//   the variance of pre-activations roughly constant across
//   layers at initialisation, avoiding vanishing or exploding
//   activations in the early training steps. Uses the
//   standard library's rand() which was seeded once in the
//   FraudDetector constructor.
// ----------------------------------------------------------
static double xavier(int fan_in, int fan_out)
{
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    // rand() gives [0, RAND_MAX]; map to [-limit, +limit]
    double r = static_cast<double>(std::rand()) / RAND_MAX;
    return -limit + 2.0 * limit * r;
}

// ----------------------------------------------------------
// NeuralNet (constructor)
//
// Short:
//   Allocate and Xavier-initialise the 2-layer MLP.
//
// Detailed:
//   Sizes W1 as hidden x input and W2 as hidden x 1 (one
//   output neuron). Every weight is drawn with xavier();
//   both bias vectors stay at zero. The z1_/a1_ activation
//   buffers are also allocated here so they can be reused by
//   every forward/backward pass without further allocation.
// ----------------------------------------------------------
NeuralNet::NeuralNet(int input_size, int hidden_size)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      b2_(0.0),
      z2_(0.0),
      a2_(0.0)
{
    // Initialise W1 with Xavier, b1 to zero
    W1_.resize(hidden_size_, std::vector<double>(input_size_, 0.0));
    b1_.resize(hidden_size_, 0.0);
    for (int j = 0; j < hidden_size_; ++j) {
        for (int i = 0; i < input_size_; ++i) {
            W1_[j][i] = xavier(input_size_, hidden_size_);
        }
    }

    // Initialise W2 with Xavier, b2 to zero
    W2_.resize(hidden_size_, 0.0);
    for (int j = 0; j < hidden_size_; ++j) {
        W2_[j] = xavier(hidden_size_, 1);
    }

    // Resize cached activation buffers
    z1_.resize(hidden_size_, 0.0);
    a1_.resize(hidden_size_, 0.0);
}

// ----------------------------------------------------------
// sigmoid
//
// Short:
//   Numerically stable logistic sigmoid.
//
// Detailed:
//   Returns 1 / (1 + exp(-x)), but clamps x to the safe range
//   [-20, 20] before calling std::exp. Outside that range the
//   result is indistinguishable from 0 or 1 in double
//   precision, and clamping avoids the risk of exp() overflow
//   on extreme inputs during early training.
// ----------------------------------------------------------
double NeuralNet::sigmoid(double x) const
{
    // Clamp to avoid overflow in exp()
    if (x >  20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

// ----------------------------------------------------------
// forward
//
// Short:
//   Run one sample through the MLP and return its probability.
//
// Detailed:
//   Implements the two-layer feed-forward pass:
//     z1[j] = b1[j] + sum_i W1[j][i] * x[i]
//     a1[j] = ReLU(z1[j])
//     z2    = b2    + sum_j W2[j]    * a1[j]
//     a2    = sigmoid(z2)  -> fraud probability in (0, 1)
//   Throws if x is not the configured input_size. Side-effect:
//   caches z1_, a1_, z2_ and a2_ so that backward_step can
//   reuse them to compute analytic gradients without a second
//   forward pass.
// ----------------------------------------------------------
double NeuralNet::forward(const std::vector<double>& x)
{
    // Safety check
    if (static_cast<int>(x.size()) != input_size_) {
        throw std::runtime_error("NeuralNet::forward - input size mismatch");
    }

    // Layer 1: z1[j] = b1[j] + sum_i(W1[j][i] * x[i])
    for (int j = 0; j < hidden_size_; ++j) {
        double z = b1_[j];
        for (int i = 0; i < input_size_; ++i) {
            z += W1_[j][i] * x[i];
        }
        z1_[j] = z;
        a1_[j] = relu(z);
    }

    // Layer 2: z2 = b2 + sum_j(W2[j] * a1[j])
    double z2 = b2_;
    for (int j = 0; j < hidden_size_; ++j) {
        z2 += W2_[j] * a1_[j];
    }
    z2_ = z2;
    a2_ = sigmoid(z2);

    return a2_;
}

// ----------------------------------------------------------
// backward_step
//
// Short:
//   Apply one SGD weight update from a single (x, y) sample.
//
// Detailed:
//   Runs forward(x) to populate the activation cache, then
//   walks backward through the network using the analytic
//   gradients of binary cross-entropy composed with sigmoid.
//
//   The output layer collapses neatly to:
//       dL/dz2 = a2 - y_true
//   which feeds updates for W2 and b2. Propagating back
//   through the ReLU hidden layer:
//       dL/da1[j]   = dL/dz2 * W2[j]
//       dL/dz1[j]   = dL/da1[j] * relu'(z1[j])
//       dL/dW1[j,i] = dL/dz1[j] * x[i]
//       dL/db1[j]   = dL/dz1[j]
//   Every parameter is updated in place as  W <- W - lr * dL/dW.
// ----------------------------------------------------------
void NeuralNet::backward_step(const std::vector<double>& x,
                              double y_true,
                              double lr)
{
    // Run forward pass to populate cached values
    forward(x);

    // Output layer gradient
    double dz2 = a2_ - y_true;   // dL/dz2

    // Update W2 and b2
    for (int j = 0; j < hidden_size_; ++j) {
        double grad_W2j = dz2 * a1_[j];
        W2_[j] -= lr * grad_W2j;
    }
    b2_ -= lr * dz2;

    // Hidden layer gradients
    for (int j = 0; j < hidden_size_; ++j) {
        // dL/da1[j] = dL/dz2 * W2[j]
        double da1j = dz2 * W2_[j];
        // dL/dz1[j] = dL/da1[j] * relu'(z1[j])
        double dz1j = da1j * relu_deriv(z1_[j]);

        // Update W1[j][*] and b1[j]
        for (int i = 0; i < input_size_; ++i) {
            W1_[j][i] -= lr * dz1j * x[i];
        }
        b1_[j] -= lr * dz1j;
    }
}

// ----------------------------------------------------------
// train
//
// Short:
//   Run `epochs` passes of online SGD over the dataset.
//
// Detailed:
//   For every epoch and every sample, computes the current
//   binary cross-entropy loss (with p clamped to [1e-7,
//   1-1e-7] to avoid log(0)) for monitoring, then performs
//   one backward_step weight update. The order is not
//   shuffled between epochs, which is fine for a small,
//   well-behaved dataset. Prints the average loss at epoch 1
//   and every 20 epochs thereafter so the caller can watch it
//   decrease. Does nothing on an empty dataset except log a
//   warning.
// ----------------------------------------------------------
void NeuralNet::train(std::vector<std::vector<double>>& X,
                      std::vector<double>& y,
                      int epochs,
                      double lr)
{
    int N = static_cast<int>(X.size());
    if (N == 0) {
        std::cerr << "[NeuralNet] Warning: empty training set.\n";
        return;
    }

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;

        for (int s = 0; s < N; ++s) {
            // Compute current prediction (for loss logging)
            double p = forward(X[s]);
            // Clamp to avoid log(0)
            double p_clamp = std::max(1e-7, std::min(1.0 - 1e-7, p));
            total_loss += -y[s] * std::log(p_clamp)
                          - (1.0 - y[s]) * std::log(1.0 - p_clamp);

            // SGD update
            backward_step(X[s], y[s], lr);
        }

        // Print average loss every 20 epochs
        if (epoch % 20 == 0 || epoch == 1) {
            std::cout << "[Epoch " << epoch << "/" << epochs
                      << "]  avg loss = "
                      << (total_loss / N) << "\n";
        }
    }
    std::cout << "[NeuralNet] Training complete.\n";
}

// ----------------------------------------------------------
// save_weights
//
// Short:
//   Serialise all model parameters to a human-readable text file.
//
// Detailed:
//   File layout (space / newline separated):
//     line 1 : input_size hidden_size
//     next hidden_size lines : rows of W1 (one per hidden unit)
//     next line              : b1 vector
//     next line              : W2 vector
//     next line              : b2 scalar
//   Overwrites the file if it exists. Failures to open the
//   file are reported to stderr but not propagated — the
//   caller just sees that no file was written.
// ----------------------------------------------------------
void NeuralNet::save_weights(const std::string& filename) const
{
    std::ofstream f(filename);
    if (!f) {
        std::cerr << "[NeuralNet] Cannot open file for writing: " << filename << "\n";
        return;
    }
    f << input_size_ << " " << hidden_size_ << "\n";

    // W1
    for (int j = 0; j < hidden_size_; ++j) {
        for (int i = 0; i < input_size_; ++i) {
            f << W1_[j][i];
            if (i + 1 < input_size_) f << " ";
        }
        f << "\n";
    }
    // b1
    for (int j = 0; j < hidden_size_; ++j) {
        f << b1_[j];
        if (j + 1 < hidden_size_) f << " ";
    }
    f << "\n";

    // W2
    for (int j = 0; j < hidden_size_; ++j) {
        f << W2_[j];
        if (j + 1 < hidden_size_) f << " ";
    }
    f << "\n";

    // b2
    f << b2_ << "\n";
    std::cout << "[NeuralNet] Weights saved to " << filename << "\n";
}

// ----------------------------------------------------------
// load_weights
//
// Short:
//   Read model parameters back from a save_weights file.
//
// Detailed:
//   Expects the exact layout produced by save_weights. First
//   reads input_size and hidden_size and refuses to proceed if
//   they do not match the current model — so callers must
//   construct the NeuralNet with matching dimensions before
//   loading. On success, every weight and bias is overwritten
//   in place; the cached activations are left untouched.
// ----------------------------------------------------------
void NeuralNet::load_weights(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f) {
        std::cerr << "[NeuralNet] Cannot open file for reading: " << filename << "\n";
        return;
    }

    int in_sz, hid_sz;
    f >> in_sz >> hid_sz;
    if (in_sz != input_size_ || hid_sz != hidden_size_) {
        std::cerr << "[NeuralNet] Weight file dimensions don't match model.\n";
        return;
    }

    for (int j = 0; j < hidden_size_; ++j)
        for (int i = 0; i < input_size_; ++i)
            f >> W1_[j][i];

    for (int j = 0; j < hidden_size_; ++j) f >> b1_[j];
    for (int j = 0; j < hidden_size_; ++j) f >> W2_[j];
    f >> b2_;

    std::cout << "[NeuralNet] Weights loaded from " << filename << "\n";
}
