//
// Created by Chi Zhang on 5/12/22.
//

#ifndef TPDS22_CODE_TRANSITION_H
#define TPDS22_CODE_TRANSITION_H

#include <torch/torch.h>
#include <map>
#include <string>

typedef std::map<std::string, torch::Tensor> Transition;

struct EnvSpec {
    EnvSpec(std::vector<int64_t> obs_shape,
            torch::ScalarType obs_dtype,
            std::vector<int64_t> act_shape,
            torch::ScalarType act_dtype,
            double max_act) :
            obs_shape(std::move(obs_shape)),
            act_shape(std::move(act_shape)),
            obs_dtype(obs_dtype),
            act_dtype(act_dtype),
            max_act(max_act) {
    }

    std::vector<int64_t> obs_shape;
    std::vector<int64_t> act_shape;
    c10::ScalarType obs_dtype;
    c10::ScalarType act_dtype;
    double max_act;
};

#endif //TPDS22_CODE_TRANSITION_H
