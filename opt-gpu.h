#ifndef OPT_GPU_H
#define OPT_GPU_H

#include "la/la-gpu.h"

namespace opt {

    namespace gpu {

        void const_step_update(la::gpu::vector_like<double>& theta,
            la::gpu::vector_like<double> const& loss_grad,
            double step_size);

        void const_step_update(la::gpu::matrix_like<double>& theta,
            la::gpu::matrix_like<double> const& loss_grad,
            double step_size);

        void const_step_update(la::gpu::tensor_like<double>& theta,
            la::gpu::tensor_like<double> const& loss_grad,
            double step_size);

        void const_step_update_momentum(la::gpu::vector_like<double>& theta,
            la::gpu::vector_like<double> const& loss_grad,
            la::gpu::vector_like<double>& update,
            double momentum,
            double step_size);

        void const_step_update_momentum(la::gpu::matrix_like<double>& theta,
            la::gpu::matrix_like<double> const& loss_grad,
            la::gpu::matrix_like<double>& update,
            double momentum,
            double step_size);

        void const_step_update_momentum(la::gpu::tensor_like<double>& theta,
            la::gpu::tensor_like<double> const& loss_grad,
            la::gpu::tensor_like<double>& update,
            double momentum,
            double step_size);

        void adagrad_update(la::gpu::vector_like<double>& theta,
            la::gpu::vector_like<double> const& loss_grad,
            la::gpu::vector_like<double>& accu_grad_sq,
            double step_size);

        void adagrad_update(la::gpu::matrix_like<double>& theta,
            la::gpu::matrix_like<double> const& loss_grad,
            la::gpu::matrix_like<double>& accu_grad_sq,
            double step_size);

        void adagrad_update(la::gpu::tensor_like<double>& theta,
            la::gpu::tensor_like<double> const& loss_grad,
            la::gpu::tensor_like<double>& accu_grad_sq,
            double step_size);

        void rmsprop_update(la::gpu::vector_like<double>& theta,
            la::gpu::vector_like<double> const& loss_grad,
            la::gpu::vector_like<double>& accu_grad_sq,
            double decay,
            double step_size);

        void rmsprop_update(la::gpu::matrix_like<double>& theta,
            la::gpu::matrix_like<double> const& loss_grad,
            la::gpu::matrix_like<double>& accu_grad_sq,
            double decay,
            double step_size);

        void rmsprop_update(la::gpu::tensor_like<double>& theta,
            la::gpu::tensor_like<double> const& loss_grad,
            la::gpu::tensor_like<double>& accu_grad_sq,
            double decay,
            double step_size);

        void adam_update(la::gpu::vector_like<double>& theta,
            la::gpu::vector_like<double> const& loss_grad,
            la::gpu::vector_like<double>& first_moment,
            la::gpu::vector_like<double>& second_moment,
            int& time, double alpha, double beta1, double beta2);

        void adam_update(la::gpu::matrix_like<double>& theta,
            la::gpu::matrix_like<double> const& loss_grad,
            la::gpu::matrix_like<double>& first_moment,
            la::gpu::matrix_like<double>& second_moment,
            int& time, double alpha, double beta1, double beta2);

        void adam_update(la::gpu::tensor_like<double>& theta,
            la::gpu::tensor_like<double> const& loss_grad,
            la::gpu::tensor_like<double>& first_moment,
            la::gpu::tensor_like<double>& second_moment,
            int& time, double alpha, double beta1, double beta2);

    }

}

#endif
