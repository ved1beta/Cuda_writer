#pragma once
#include <vector>
#include <string>

// ============================================================
// neural_net.h - 2-layer Multi-Layer Perceptron (MLP)
// ============================================================
// Architecture:
//   Input  (11 features)
//     |
//   Hidden (8 neurons)  -- ReLU activation
//     |
//   Output (1 neuron)   -- Sigmoid activation  -> fraud probability
//
// Training: stochastic gradient descent (SGD) with binary
// cross-entropy loss.  Gradients are computed analytically.

class NeuralNet {
public:
    // -------------------------------------------------------
    // NeuralNet (constructor)
    //
    // Short:
    //   Build a 2-layer MLP with Xavier-uniform weights.
    //
    // Detailed:
    //   input_size defaults to 11 (the GNN enriched feature
    //   vector) and hidden_size to 8. Both bias vectors start
    //   at zero; W1 and W2 are filled by Xavier samples.
    //   Activation caches are allocated once so no further
    //   heap work is needed during training.
    // -------------------------------------------------------
    NeuralNet(int input_size = 11, int hidden_size = 8);

    // -------------------------------------------------------
    // forward
    //
    // Short:
    //   Compute fraud probability for one input sample.
    //
    // Detailed:
    //   Feeds x through  Dense+ReLU  ->  Dense+Sigmoid  and
    //   returns a scalar in (0, 1). Caches z1_/a1_/z2_/a2_
    //   internally so backward_step can reuse them without a
    //   second forward pass. Throws if x's size does not match
    //   input_size.
    // -------------------------------------------------------
    double forward(const std::vector<double>& x);

    // -------------------------------------------------------
    // train
    //
    // Short:
    //   Run online SGD for `epochs` passes.
    //
    // Detailed:
    //   X : feature matrix (N x input_size)
    //   y : labels (N values, each 0.0 or 1.0)
    //   epochs : number of full passes over X
    //   lr     : SGD learning rate
    //   Logs the mean binary cross-entropy loss at epoch 1 and
    //   every 20 epochs so training progress is visible.
    //   Updates are applied sample-by-sample with
    //   backward_step; the dataset order is not reshuffled
    //   between epochs.
    // -------------------------------------------------------
    void train(std::vector<std::vector<double>>& X,
               std::vector<double>& y,
               int epochs = 200,
               double lr  = 0.05);

    // -------------------------------------------------------
    // backward_step
    //
    // Short:
    //   Apply one SGD update from a single (x, y_true) sample.
    //
    // Detailed:
    //   Runs forward(x) to populate the activation cache, then
    //   uses the analytic sigmoid + binary-cross-entropy
    //   gradient  dL/dz2 = a2 - y_true  and ReLU backprop to
    //   update every weight and bias in place by subtracting
    //   lr * gradient.
    // -------------------------------------------------------
    void backward_step(const std::vector<double>& x,
                       double y_true,
                       double lr);

    // -------------------------------------------------------
    // save_weights
    //
    // Short:
    //   Write all parameters to a plain-text file.
    //
    // Detailed:
    //   Format: "input_size hidden_size" on line 1, then W1
    //   (hidden_size rows), b1, W2 and b2 on subsequent lines.
    //   The file is human-readable so small models can be
    //   diffed or hand-edited. Overwrites `filename` if it
    //   already exists.
    // -------------------------------------------------------
    void save_weights(const std::string& filename) const;

    // -------------------------------------------------------
    // load_weights
    //
    // Short:
    //   Load parameters previously written by save_weights.
    //
    // Detailed:
    //   Expects the exact file format produced by
    //   save_weights. Refuses to proceed if the stored
    //   input_size/hidden_size differ from the current model,
    //   so callers must construct a NeuralNet with matching
    //   dimensions before loading.
    // -------------------------------------------------------
    void load_weights(const std::string& filename);

    // -------------------------------------------------------
    // get_input_size
    //
    // Short:
    //   Return the configured input dimensionality.
    //
    // Detailed:
    //   Exposed so GNN can sanity-check that its enriched
    //   feature vector matches what the MLP expects.
    // -------------------------------------------------------
    int get_input_size()  const { return input_size_;  }

    // -------------------------------------------------------
    // get_hidden_size
    //
    // Short:
    //   Return the configured hidden-layer width.
    //
    // Detailed:
    //   Mainly useful for diagnostic output; not required for
    //   normal training or inference.
    // -------------------------------------------------------
    int get_hidden_size() const { return hidden_size_; }

private:
    int input_size_;
    int hidden_size_;

    // Layer 1 weights: W1[hidden][input]
    std::vector<std::vector<double>> W1_;
    // Layer 1 biases: b1[hidden]
    std::vector<double> b1_;

    // Layer 2 weights: W2[hidden]  (single output neuron)
    std::vector<double> W2_;
    // Layer 2 bias
    double b2_;

    // Cached activations from the last forward pass
    // (needed by backward_step)
    std::vector<double> z1_;   // pre-activation hidden
    std::vector<double> a1_;   // post-activation hidden (ReLU)
    double              z2_;   // pre-activation output
    double              a2_;   // output (sigmoid)

    // -------------------------------------------------------
    // relu
    //
    // Short:
    //   Rectified Linear Unit hidden-layer activation.
    //
    // Detailed:
    //   Returns x for positive inputs and 0 otherwise. Chosen
    //   over sigmoid/tanh for the hidden layer because it
    //   avoids the vanishing-gradient problem during backprop.
    // -------------------------------------------------------
    double relu      (double x) const { return x > 0.0 ? x : 0.0; }

    // -------------------------------------------------------
    // sigmoid
    //
    // Short:
    //   Numerically stable logistic sigmoid for the output.
    //
    // Detailed:
    //   Squashes the raw output z2 into a probability in
    //   (0, 1). The implementation clamps |x| at 20 before
    //   calling std::exp so extreme inputs cannot overflow.
    // -------------------------------------------------------
    double sigmoid   (double x) const;

    // -------------------------------------------------------
    // relu_deriv
    //
    // Short:
    //   Derivative of ReLU w.r.t. its input.
    //
    // Detailed:
    //   Returns 1 for positive pre-activations and 0 otherwise.
    //   Used during backward_step when propagating gradients
    //   through the hidden layer.
    // -------------------------------------------------------
    double relu_deriv(double x) const { return x > 0.0 ? 1.0 : 0.0; }
};
