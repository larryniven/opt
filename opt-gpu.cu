#include "opt/opt-gpu.h"
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <cuda_runtime.h>

namespace opt {

    namespace gpu {

        struct iadagrad_update_op {
            double step_size;

            template <class T>
            __host__ __device__
            void operator()(T t) const
            {
                auto& theta = thrust::get<0>(t);
                auto& loss_grad = thrust::get<1>(t);
                auto& accu_grad_sq = thrust::get<2>(t);

                accu_grad_sq += pow(loss_grad, 2);

                if (accu_grad_sq > 0) {
                    theta -= loss_grad * step_size
                        / sqrt(accu_grad_sq);
                }
            }
        };

        void adagrad_update(la::gpu::vector_like<double>& theta,
            la::gpu::vector_like<double> const& loss_grad,
            la::gpu::vector_like<double>& accu_grad_sq,
            double step_size)
        {
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(theta.begin()),
                    thrust::device_ptr<double const>(loss_grad.begin()),
                    thrust::device_ptr<double>(accu_grad_sq.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(theta.end()),
                    thrust::device_ptr<double const>(loss_grad.end()),
                    thrust::device_ptr<double>(accu_grad_sq.end()))),
                iadagrad_update_op { step_size });
        }

        void adagrad_update(la::gpu::matrix_like<double>& theta,
            la::gpu::matrix_like<double> const& loss_grad,
            la::gpu::matrix_like<double>& accu_grad_sq,
            double step_size)
        {
            unsigned int size = theta.rows() * theta.cols();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(theta.data()),
                    thrust::device_ptr<double const>(loss_grad.data()),
                    thrust::device_ptr<double>(accu_grad_sq.data()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(theta.data() + size),
                    thrust::device_ptr<double const>(loss_grad.data() + size),
                    thrust::device_ptr<double>(accu_grad_sq.data() + size))),
                iadagrad_update_op { step_size });
        }

        struct iadam_update_op {
            double time;
            double alpha;
            double beta1;
            double beta2;
            double b1;
            double b2;

            template <class T>
            __host__ __device__
            void operator()(T t) const
            {
                auto& theta = thrust::get<0>(t);
                auto& loss_grad = thrust::get<1>(t);
                auto& first_moment = thrust::get<2>(t);
                auto& second_moment = thrust::get<3>(t);

                first_moment = first_moment * beta1 + loss_grad * (1 - beta1);
                second_moment = second_moment * beta2 + pow(loss_grad, 2) * (1 - beta2);

                theta -= alpha * first_moment / b1
                    / (std::sqrt(second_moment / b2) + 1e-8);
            }
        };

        void adam_update(la::gpu::vector_like<double>& theta,
            la::gpu::vector_like<double> const& loss_grad,
            la::gpu::vector_like<double>& first_moment,
            la::gpu::vector_like<double>& second_moment,
            double time, double alpha, double beta1, double beta2)
        {
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(theta.begin()),
                    thrust::device_ptr<double const>(loss_grad.begin()),
                    thrust::device_ptr<double>(first_moment.begin()),
                    thrust::device_ptr<double>(second_moment.begin()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(theta.end()),
                    thrust::device_ptr<double const>(loss_grad.end()),
                    thrust::device_ptr<double>(first_moment.end()),
                    thrust::device_ptr<double>(second_moment.end()))),
                iadam_update_op { time, alpha, beta1, beta2, 1 - pow(beta1, time), 1 - pow(beta2, time) });
        }

        void adam_update(la::gpu::matrix_like<double>& theta,
            la::gpu::matrix_like<double> const& loss_grad,
            la::gpu::matrix_like<double>& first_moment,
            la::gpu::matrix_like<double>& second_moment,
            double time, double alpha, double beta1, double beta2)
        {
            unsigned int size = theta.rows() * theta.cols();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(theta.data()),
                    thrust::device_ptr<double const>(loss_grad.data()),
                    thrust::device_ptr<double>(first_moment.data()),
                    thrust::device_ptr<double>(second_moment.data()))),
                thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_ptr<double>(theta.data() + size),
                    thrust::device_ptr<double const>(loss_grad.data() + size),
                    thrust::device_ptr<double>(first_moment.data() + size),
                    thrust::device_ptr<double>(second_moment.data() + size))),
                iadam_update_op { time, alpha, beta1, beta2, 1 - pow(beta1, time), 1 - pow(beta2, time) });
        }

    }
}

