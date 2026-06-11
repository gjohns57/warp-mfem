from typing import Any, Callable

import warp as wp

from mfem.accumulate import TiledAccumulator
from mfem.utils import linear_accumulate


@wp.kernel
def _project_search_kernel(
    original_pose: wp.array[Any],
    search_direction: wp.array[Any],
    alpha: wp.array[wp.float32],
    new_pose: wp.array[Any],
):
    tid = wp.tid()
    new_pose[tid] = original_pose[tid] + alpha[0] * search_direction[tid]


@wp.kernel
def _condition_kernel(
    objective: wp.array[wp.float32],
    alpha_and_t: wp.array[wp.float32],
    tau: wp.float32,
    threshold: wp.float32,
    max_iter: wp.int32,
    cond_and_iteration: wp.array[wp.int32],
):
    cond_and_iteration[0] = wp.where(
        (objective[0] - objective[1]) + threshold < alpha_and_t[0] * alpha_and_t[1]
        and cond_and_iteration[1] < max_iter,
        1,
        0,
    )

    # print(objective[0])
    # print(objective[1])

    cond_and_iteration[1] = cond_and_iteration[1] + 1
    alpha_and_t[0] = alpha_and_t[0] * tau


@wp.kernel
def _iteration_condition(
    cond_and_iteration: wp.array[wp.int32],
    max_iter: wp.int32,
):
    cond_and_iteration[0] = wp.where(cond_and_iteration[1] < max_iter, 1, 0)


def backtracking_line_search(
    objective_fn: Callable[[Any, tuple[wp.array], wp.array], None],
    dof_arrays: tuple[wp.array],
    search_direction: tuple[wp.array],
    gradient: tuple[wp.array],
    accumulator: TiledAccumulator,
    max_iterations=10,
    alpha0=1.0,
    tau=0.5,
    c=0.5,
    threshold=1e-4,
    use_cuda_graph=True,
) -> wp.array[wp.float32]:
    # Initial energy at starting state
    objective = wp.zeros(2, dtype=wp.float32)
    cond_and_iteration = wp.zeros(2, wp.int32)
    objective_fn(dof_arrays, objective[0:1])

    # This will be t * alpha_i where t = -c * m
    alpha_and_t = wp.array([alpha0, 0.0], dtype=wp.float32)
    new_dof: tuple[wp.array] = tuple(map(lambda array: wp.clone(array), dof_arrays))

    for i in range(len(dof_arrays)):
        accumulator.compute_dot(
            gradient[i].view(wp.float32).flatten(),
            search_direction[i].view(wp.float32).flatten(),
        )
        linear_accumulate(alpha_and_t[1:2], accumulator.result(), beta=-c)

    def backtracking_loop():
        for i in range(len(dof_arrays)):
            wp.launch(
                _project_search_kernel,
                dim=new_dof[i].shape,
                inputs=[dof_arrays[i], search_direction[i], alpha_and_t[0:2]],
                outputs=[new_dof[i]],
            )

        objective_fn(new_dof, objective[1:2])

        wp.launch(
            _condition_kernel,
            dim=1,
            inputs=[objective, alpha_and_t, tau, threshold, max_iterations],
            outputs=[cond_and_iteration],
        )

    backtracking_loop()

    wp.capture_while(
        cond_and_iteration[0:1],
        while_body=backtracking_loop,
    )

    wp.launch(_iteration_condition, dim=1, inputs=[cond_and_iteration, max_iterations])

    def write_udpate():
        for i in range(len(dof_arrays)):
            wp.copy(dof_arrays[i], new_dof[i])

    wp.capture_if(
        cond_and_iteration[0:1],
        on_true=write_udpate,
        on_false=(lambda: print("Line search failed")),
    )

    return objective
