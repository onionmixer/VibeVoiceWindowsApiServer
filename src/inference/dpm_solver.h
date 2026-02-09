#pragma once
#include <cstdint>
#include <string>
#include <vector>

class DPMSolver {
public:
    // betaSchedule: "cosine", predictionType: "v_prediction"
    DPMSolver(int numTrainTimesteps = 1000, int numInferenceSteps = 20,
              const std::string& betaSchedule = "cosine",
              const std::string& predictionType = "v_prediction",
              int solverOrder = 2);

    // Set number of inference steps (recomputes timesteps/sigmas)
    void setTimesteps(int numInferenceSteps);

    // Get inference timestep at step index (for diffusion head input)
    int timestep(int stepIndex) const;
    int numSteps() const;

    // Single solver step. All arrays are fp32, size n.
    // modelOutput: v_prediction from diffusion head
    // sample: current noisy latent
    // stepIndex: 0-based step counter
    // prevSample: output denoised latent
    void step(const float* modelOutput, int stepIndex,
              const float* sample, float* prevSample, int n);

    // Reset solver state (call before each new generation)
    void reset();

private:
    void computeCosineSchedule();
    void computeTimesteps();

    // Convert v_prediction to x0_prediction
    void convertModelOutput(const float* modelOutput, const float* sample,
                            int stepIndex, float* x0Pred, int n);

    void firstOrderUpdate(const float* x0, int stepIndex,
                          const float* sample, float* output, int n);
    void secondOrderUpdate(int stepIndex,
                           const float* sample, float* output, int n);

    int numTrainTimesteps_, numInferenceSteps_, solverOrder_;
    std::string predictionType_;

    std::vector<float> alphasCumprod_;  // [numTrainTimesteps]
    std::vector<float> sigmas_;         // [numInferenceSteps + 1]
    std::vector<float> lambdas_;        // [numInferenceSteps + 1] log SNR
    std::vector<int> timesteps_;        // [numInferenceSteps]

    // Multi-step model output history: [solverOrder][n]
    std::vector<std::vector<float>> modelOutputs_;
    int lowerOrderNums_ = 0;
};
