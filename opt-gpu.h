#ifndef OPT_GPU_H
#define OPT_GPU_H

#include "la/la-gpu.h"

namespace opt {

    namespace gpu {

        void adagrad_update(la::gpu::vector<double>& theta,
            la::gpu::vector<double> const& loss_grad,
            la::gpu::vector<double>& accu_grad_sq,
            double step_size);

        void adagrad_update(la::gpu::matrix<double>& theta,
            la::gpu::matrix<double> const& loss_grad,
            la::gpu::matrix<double>& accu_grad_sq,
            double step_size);

        void adam_update(la::gpu::vector<double>& theta,
            la::gpu::vector<double> const& loss_grad,
            la::gpu::vector<double>& first_moment,
            la::gpu::vector<double>& second_moment,
            double time, double alpha, double beta1, double beta2);

        void adam_update(la::gpu::matrix<double>& theta,
            la::gpu::matrix<double> const& loss_grad,
            la::gpu::matrix<double>& first_moment,
            la::gpu::matrix<double>& second_moment,
            double time, double alpha, double beta1, double beta2);

    }

}

#endif
