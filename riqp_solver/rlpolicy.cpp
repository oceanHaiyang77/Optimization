#include "rlpolicy.h"
#include <torch/script.h>
#include <Eigen/Dense>

using Module = torch::jit::script::Module;

namespace rlqp {
    struct Policy {
        Module baseModule_;
        Module module_;
    };
}

void *rl_policy_load(const char *module_path) {
    if (module_path == nullptr || module_path[0] == '\0')
        return nullptr;

    rlqp::Policy* policy = new rlqp::Policy();
    try {
        policy->baseModule_ = torch::jit::load(module_path);
        policy->module_ = torch::jit::optimize_for_inference(policy->baseModule_);
        std::clog << "Loaded model " << module_path << std::endl;
        return policy;
    } catch (const c10::Error& ex) {
        std::clog << "error loading model: " << ex.what() << std::endl;
        delete policy;
        return nullptr;
    }
}

int rl_policy_unload(void *ptr) {
    rlqp::Policy *policy = static_cast<rlqp::Policy*>(ptr);
    if (policy)
        delete policy;
    return 0;
}

int rl_policy_compute_vec(OSQPWorkspace* work) {
    using namespace Eigen;

    rlqp::Policy *policy = static_cast<rlqp::Policy*>(work->rl_rho_policy);
    if (policy == nullptr)
        return 1;

    static constexpr int stride = 6;
    using QPVec = Array<c_float, Eigen::Dynamic, 1>;
    using RLVec = Array<float, Eigen::Dynamic, 1>;
    using InputVec = Array<float, Eigen::Dynamic, stride, Eigen::RowMajor | Eigen::DontAlign>;

    int m = (int)work->data->m;
    
    // create Eigen wrappers for the vectors (these are zero-copy)
    Map<QPVec> l(work->data->l, m);
    Map<QPVec> u(work->data->u, m);
    Map<QPVec> z(work->z, m);
    Map<QPVec> y(work->y, m);
    Map<QPVec> Ax(work->Ax, m);
    Map<QPVec> rhoVec(work->rho_vec, m);

    // construct a tensor by stacking the vectors
    // InputVec inputData(m, stride);
    // inputData.col(0) = rhoVec.template cast<float>();
    // inputData.col(1) = l.template cast<float>();
    // inputData.col(2) = u.template cast<float>();
    // inputData.col(3) = z.template cast<float>();
    // inputData.col(4) = y.template cast<float>();
    // inputData.col(5) = Ax.template cast<float>();

    // then move/copy into a tensor
    // at::Tensor piInputs = torch::from_blob(
    //     inputData.data(), {m, stride},
    //     torch::dtype(torch::kFloat32).requires_grad(false));

    // Alternate route--this seems like it should be more efficient.
    // Allocate uninitialized tensor, then copy in the data (once)
    // TODO: preallocate in the policy.
    at::Tensor piInputs = torch::empty(
        {m, stride},
        torch::dtype(torch::kFloat32).requires_grad(false));

    Map<InputVec> inputData(static_cast<float*>(piInputs.data_ptr()), m, stride);
    inputData.col(0) = Ax.template cast<float>();
    inputData.col(1) = y.template cast<float>();
    inputData.col(2) = z.template cast<float>();
    inputData.col(3) = l.template cast<float>();
    inputData.col(4) = u.template cast<float>();
    inputData.col(5) = rhoVec.template cast<float>();

    // Compute the policy value based on the input
    // TODO: we could also cache the argument to forward
    // (std::vector<IValue> with a single tensor element.)
    at::Tensor piOutput = policy->module_.forward({{piInputs}}).toTensor();

    // Store (after casting) the values to the rho vector.
    rhoVec = Map<RLVec>(piOutput.data_ptr<float>(), m).template cast<c_float>();
    
    assert(rhoVec.allFinite()); // TODO: this is useful for debugging, but should probably be replaced for regular usage

    Map<QPVec>(work->rho_inv_vec, m) = 1 / rhoVec;
    
    return 0;
}
