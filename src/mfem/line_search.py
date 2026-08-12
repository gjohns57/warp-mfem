from typing import Any, Callable

import warp as wp
import warp.sparse as ws

from mfem.accumulate import TiledAccumulator
from mfem.utils import linear_accumulate

from .types import float_type


@wp.kernel
def _reset_line_search_state(
    alpha_and_t: wp.array(dtype=float_type),
    cond_and_iteration: wp.array(dtype=wp.int32),
    alpha0: float_type,
):
    alpha_and_t[0] = alpha0
    alpha_and_t[1] = float_type(0.0)
    cond_and_iteration[0] = wp.int32(0)
    cond_and_iteration[1] = wp.int32(0)


@wp.kernel
def _project_search_kernel(
    original_pose: wp.array[Any],
    search_direction: wp.array[Any],
    alpha: wp.array[float_type],
    new_pose: wp.array[Any],
):
    tid = wp.tid()
    new_pose[tid] = original_pose[tid] + alpha[0] * search_direction[tid]


@wp.kernel
def _condition_kernel(
    objective: wp.array[float_type],
    alpha_and_t: wp.array[float_type],
    tau: float_type,
    threshold: float_type,
    max_iter: wp.int32,
    cond_and_iteration: wp.array[wp.int32],
):
    cond_and_iteration[0] = wp.where(
        (objective[0] - objective[1]) + threshold < (alpha_and_t[0] * (alpha_and_t[1]))
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


@wp.kernel
def _add_noise(
    array: wp.array[wp.float32],
    scale: wp.float32,
    state: wp.uint32,
):
    tid = wp.tid()
    array[tid] += (2.0 * wp.randf(state) - 1.0) * scale


def backtracking_line_search(
    objective_fn: Callable[[Any, tuple[wp.array], wp.array], None],
    dof_arrays: tuple[wp.array],
    search_direction: tuple[wp.array],
    gradient: tuple[wp.array],
    hessians: tuple[ws.BsrMatrix],
    accumulator: TiledAccumulator,
    max_iterations=10,
    alpha0=1.0,
    tau=0.5,
    c=0.5,
    threshold=1e-10,
) -> wp.array[float_type]:
    # Initial energy at starting state
    objective = wp.zeros(2, dtype=float_type)
    cond_and_iteration = wp.zeros(2, wp.int32)

    # This will be t * alpha_i where t = -c * m
    alpha_and_t = wp.array([alpha0, 0.0], dtype=float_type)
    new_dof: tuple[wp.array] = tuple(map(lambda array: wp.clone(array), dof_arrays))

    # Reset mutable state so graph replays start fresh each step.
    # alpha_and_t[0] shrinks via tau each iteration and must be restored to alpha0.
    # alpha_and_t[1] uses linear_accumulate (additive) and must be zeroed before
    # the dot-product loop below so it doesn't grow across replays.
    # cond_and_iteration[1] (iteration counter) must be zeroed so it doesn't
    # accumulate past max_iterations and cause wp.capture_if to skip write_update.
    wp.launch(
        _reset_line_search_state,
        dim=1,
        inputs=[alpha_and_t, cond_and_iteration, alpha0],
    )

    objective_fn(dof_arrays, objective[0:1])

    for i in range(len(dof_arrays)):
        accumulator.compute_dot(
            gradient[i],
            search_direction[i],
        )
        linear_accumulate(alpha_and_t[1:2], accumulator.result(), beta=-c)

        # accumulator.compute_dot(search_direction[i], hessians[i] @ search_direction[i])
        # linear_accumulate(alpha_and_t[2:3], accumulator.result(), beta=-0.5 * c)

    def backtracking_loop():
        for i in range(len(dof_arrays)):
            wp.launch(
                _project_search_kernel,
                dim=new_dof[i].shape,
                inputs=[dof_arrays[i], search_direction[i], alpha_and_t[0:2]],
                outputs=[new_dof[i]],
            )

        objective_fn(new_dof, objective[1:2])

        # print(objective.numpy())
        # print(alpha_and_t.numpy().prod())

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

    def write_update():
        # print(
        #     f"Wrote update after {cond_and_iteration.numpy()[1]} line search iterations"
        # )
        for i in range(len(dof_arrays)):
            wp.copy(dof_arrays[i], new_dof[i])

    # write_update()
    wp.capture_if(
        cond_and_iteration[0:1],
        on_true=write_update,
    )

    return objective
