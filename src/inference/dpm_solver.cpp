#include "inference/dpm_solver.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DPMSolver::DPMSolver(int numTrainTimesteps, int numInferenceSteps,
                     const std::string& /*betaSchedule*/,
                     const std::string& predictionType,
                     int solverOrder)
    : numTrainTimesteps_(numTrainTimesteps)
    , numInferenceSteps_(numInferenceSteps)
    , solverOrder_(solverOrder)
    , predictionType_(predictionType)
{
    computeCosineSchedule();
    computeTimesteps();
}

void DPMSolver::computeCosineSchedule() {
    // Cosine schedule: alpha_bar(t) = cos((t/T + s) / (1 + s) * pi/2)^2
    // where s = 0.008
    alphasCumprod_.resize(numTrainTimesteps_);
    const float s = 0.008f;

    for (int t = 0; t < numTrainTimesteps_; ++t) {
        float progress = (float)(t + 1) / (float)numTrainTimesteps_;
        float cosVal = cosf(((progress + s) / (1.0f + s)) * (float)(M_PI / 2.0));
        alphasCumprod_[t] = cosVal * cosVal;
    }

    // Clip to avoid numerical issues
    for (auto& ab : alphasCumprod_) {
        ab = std::max(ab, 0.0001f);
        ab = std::min(ab, 0.9999f);
    }
}

void DPMSolver::computeTimesteps() {
    // Compute evenly spaced timesteps for inference
    // Following diffusers DPMSolverMultistepScheduler logic:
    // timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps + 1)
    // rounded, reversed, then drop last

    int n = numInferenceSteps_;
    std::vector<float> linspace(n + 1);
    for (int i = 0; i <= n; ++i) {
        linspace[i] = (float)i * (float)(numTrainTimesteps_ - 1) / (float)n;
    }

    // Round and reverse
    timesteps_.resize(n);
    for (int i = 0; i < n; ++i) {
        timesteps_[i] = (int)roundf(linspace[n - i]);
    }

    // Compute sigmas and lambdas at each timestep (+ final sigma)
    sigmas_.resize(n + 1);
    lambdas_.resize(n + 1);

    for (int i = 0; i < n; ++i) {
        int t = timesteps_[i];
        float ab = alphasCumprod_[t];
        float alpha_t = sqrtf(ab);
        float sigma_t = sqrtf(1.0f - ab);
        sigmas_[i] = sigma_t;
        lambdas_[i] = logf(alpha_t / sigma_t);
    }
    // Final sigma (t=0): fully denoised
    sigmas_[n] = 0.0f;
    lambdas_[n] = 1e10f;  // log(alpha/sigma) -> inf as sigma -> 0

    // Reset solver state
    reset();
}

void DPMSolver::setTimesteps(int numInferenceSteps) {
    numInferenceSteps_ = numInferenceSteps;
    computeTimesteps();
}

int DPMSolver::timestep(int stepIndex) const {
    if (stepIndex < 0 || stepIndex >= (int)timesteps_.size()) return 0;
    return timesteps_[stepIndex];
}

int DPMSolver::numSteps() const {
    return numInferenceSteps_;
}

void DPMSolver::reset() {
    modelOutputs_.clear();
    modelOutputs_.resize(solverOrder_);
    lowerOrderNums_ = 0;
}

void DPMSolver::convertModelOutput(const float* modelOutput, const float* sample,
                                   int stepIndex, float* x0Pred, int n) {
    int t = timesteps_[stepIndex];
    float ab = alphasCumprod_[t];
    float alpha_t = sqrtf(ab);
    float sigma_t = sqrtf(1.0f - ab);

    if (predictionType_ == "v_prediction") {
        // x0 = alpha_t * sample - sigma_t * v_pred
        for (int i = 0; i < n; ++i) {
            x0Pred[i] = alpha_t * sample[i] - sigma_t * modelOutput[i];
        }
    } else {
        // epsilon prediction: x0 = (sample - sigma_t * eps) / alpha_t
        for (int i = 0; i < n; ++i) {
            x0Pred[i] = (sample[i] - sigma_t * modelOutput[i]) / alpha_t;
        }
    }
}

void DPMSolver::firstOrderUpdate(const float* x0, int stepIndex,
                                  const float* sample, float* output, int n) {
    // DPM-Solver++ first order (DDIM-like)
    // x_{t-1} = (sigma_{t-1} / sigma_t) * x_t - alpha_{t-1} * (exp(-h) - 1) * D0
    // where h = lambda_{t-1} - lambda_t, D0 = x0 prediction

    int t_s = timesteps_[stepIndex];
    float ab_s = alphasCumprod_[t_s];
    float sigma_s = sqrtf(1.0f - ab_s);

    // Next step values
    float sigma_t, alpha_t;
    if (stepIndex + 1 < numInferenceSteps_) {
        int t_next = timesteps_[stepIndex + 1];
        float ab_next = alphasCumprod_[t_next];
        alpha_t = sqrtf(ab_next);
        sigma_t = sqrtf(1.0f - ab_next);
    } else {
        // Final step: fully denoised
        alpha_t = 1.0f;
        sigma_t = 0.0f;
    }

    float lambda_s = lambdas_[stepIndex];
    float lambda_t = lambdas_[stepIndex + 1];
    float h = lambda_t - lambda_s;

    // DPM-Solver++ formula:
    // x_t = (sigma_t / sigma_s) * sample + alpha_t * (1 - exp(h)) * x0
    if (sigma_t < 1e-6f) {
        // Final step: fully denoised -> output = alpha_t * x0
        for (int i = 0; i < n; ++i) {
            output[i] = alpha_t * x0[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            float term1 = (sigma_t / sigma_s) * sample[i];
            float term2 = alpha_t * (1.0f - expf(h)) * x0[i];
            output[i] = term1 + term2;
        }
    }
}

void DPMSolver::secondOrderUpdate(int stepIndex,
                                   const float* sample, float* output, int n) {
    // DPM-Solver++ second order with midpoint correction
    // Uses modelOutputs_[0] (D0, previous) and modelOutputs_[1] (D1, current)

    int s = stepIndex - 1;  // previous step

    int t_s = timesteps_[s];
    float ab_s = alphasCumprod_[t_s];
    float sigma_s = sqrtf(1.0f - ab_s);

    float sigma_t, alpha_t;
    if (stepIndex + 1 < numInferenceSteps_) {
        int t_next = timesteps_[stepIndex + 1];
        float ab_next = alphasCumprod_[t_next];
        alpha_t = sqrtf(ab_next);
        sigma_t = sqrtf(1.0f - ab_next);
    } else {
        alpha_t = 1.0f;
        sigma_t = 0.0f;
    }

    float lambda_s = lambdas_[s];
    float lambda_s1 = lambdas_[s + 1];
    float lambda_t = lambdas_[stepIndex + 1];
    float h = lambda_t - lambda_s;
    float h_0 = lambda_s1 - lambda_s;
    float r = h_0 / h;

    // D0 = modelOutputs_[0], D1 = modelOutputs_[1] (both are x0 predictions)
    const float* D0 = modelOutputs_[0].data();
    const float* D1 = modelOutputs_[1].data();

    if (sigma_t < 1e-6f) {
        // Final step: fully denoised -> output = alpha_t * D1
        for (int i = 0; i < n; ++i) {
            output[i] = alpha_t * D1[i];
        }
    } else {
        for (int i = 0; i < n; ++i) {
            float term1 = (sigma_t / sigma_s) * sample[i];
            float term2 = alpha_t * (1.0f - expf(h)) * D1[i];
            float term3 = 0.5f * alpha_t * (1.0f - expf(h)) * (1.0f / r) * (D1[i] - D0[i]);
            output[i] = term1 + term2 + term3;
        }
    }
}

void DPMSolver::step(const float* modelOutput, int stepIndex,
                     const float* sample, float* prevSample, int n) {
    // Convert model output to x0 prediction
    std::vector<float> x0(n);
    convertModelOutput(modelOutput, sample, stepIndex, x0.data(), n);

    // Store in model output history (shift buffer)
    if (solverOrder_ >= 2) {
        // Shift: [0] = [1], [1] = new
        if (modelOutputs_.size() >= 2 && !modelOutputs_[1].empty()) {
            modelOutputs_[0] = modelOutputs_[1];
        }
        modelOutputs_[1] = x0;
        // First output also goes to [0] if empty
        if (modelOutputs_[0].empty()) {
            modelOutputs_[0] = x0;
        }
    } else {
        modelOutputs_[0] = x0;
    }

    // Choose order based on history
    if (solverOrder_ >= 2 && lowerOrderNums_ >= 1 && stepIndex > 0) {
        secondOrderUpdate(stepIndex, sample, prevSample, n);
    } else {
        firstOrderUpdate(x0.data(), stepIndex, sample, prevSample, n);
    }

    lowerOrderNums_ = std::min(lowerOrderNums_ + 1, solverOrder_);
}
