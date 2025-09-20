use cubecl::{
    CubeCount, CubeDim, Runtime, client::ComputeClient, prelude::ScalarArg, server::Handle,
};

use crate::kernels::{TwiddleTransposeUniformLaunch, fft1d_r2_fused, twiddle_transpose2d_tiled};

#[derive(Debug)]
pub struct TransposeOp {
    pub rows: u32,
    pub cols: u32,
    pub twiddles: bool,
}

#[derive(Debug)]
pub enum Op {
    Fft { len: u32 },
    Transpose(TransposeOp),
}

#[derive(Debug)]
pub struct FftPlan {
    ops: Vec<Op>,
    buffer_len: usize,
}

pub struct FftParams {
    pub vec_width: u8,
    pub local_side_len: u32,
    pub threads_per_cube: u32,
}

pub fn plan_fft(dimensions: Vec<usize>, has_batch: bool, fuse_size: u32) -> FftPlan {
    let mut ops = vec![];
    let buffer_len = dimensions.iter().product::<usize>();

    for (index, &dimension) in dimensions.iter().enumerate().rev() {
        if !(index == 0 && has_batch) {
            plan1d(dimension as u32, fuse_size, &mut ops);
        }
        let rows = (buffer_len / dimension) as u32;
        if rows != 1 {
            ops.push(Op::Transpose(TransposeOp {
                rows,
                cols: dimension as u32,
                twiddles: false,
            }))
        }
    }
    FftPlan { ops, buffer_len }
}

fn plan1d(fft_len: u32, fuse_size: u32, ops: &mut Vec<Op>) {
    if fft_len <= fuse_size {
        ops.push(Op::Fft { len: fft_len });
        return;
    }

    let rest = fft_len / fuse_size;
    ops.push(Op::Transpose(TransposeOp {
        rows: fuse_size,
        cols: rest,
        twiddles: false,
    }));

    ops.push(Op::Fft { len: fuse_size });

    ops.push(Op::Transpose(TransposeOp {
        rows: rest,
        cols: fuse_size,
        twiddles: true,
    }));

    plan1d(rest, fuse_size, ops);

    ops.push(Op::Transpose(TransposeOp {
        rows: fuse_size,
        cols: rest,
        twiddles: false,
    }));
}

pub enum Direction {
    Forward,
    Inverse,
}

pub fn execute_ops<RT: Runtime>(
    array: &mut Handle,
    tmp: &mut Handle,
    plan: &FftPlan,
    direction: Direction,
    client: &ComputeClient<RT::Server, RT::Channel>,
    params: &FftParams,
) {
    let sign = match direction {
        Direction::Forward => -1.,
        Direction::Inverse => 1.,
    };
    let mut src = &array;
    let mut dst = &tmp;
    let FftPlan { ops, buffer_len } = plan;
    let FftParams {
        vec_width,
        local_side_len,
        threads_per_cube,
    } = params;

    for op in ops {
        match op {
            Op::Fft { len } => unsafe {
                let input = cubecl::prelude::ArrayArg::<RT>::from_raw_parts::<f32>(
                    &src,
                    *buffer_len,
                    *vec_width,
                );

                fft1d_r2_fused::launch_unchecked(
                    client,
                    CubeCount::new_1d((buffer_len / *len as usize) as u32),
                    CubeDim::new_1d(*threads_per_cube),
                    input,
                    ScalarArg::new(sign),
                    *len,
                );
            },
            Op::Transpose(transpose_op) => unsafe {
                let TransposeOp {
                    rows,
                    cols,
                    twiddles,
                } = transpose_op;
                let tile_size = rows * cols;
                let n_batches = *buffer_len as u32 / tile_size;
                let n_row_tiles = rows.div_ceil(*local_side_len);
                let n_col_tiles = cols.div_ceil(*local_side_len);
                let input =
                    cubecl::prelude::ArrayArg::<RT>::from_raw_parts::<f32>(src, *buffer_len, 2);

                let output =
                    cubecl::prelude::ArrayArg::<RT>::from_raw_parts::<f32>(dst, *buffer_len, 2);

                let twiddles = *twiddles as u32;
                let uniform = TwiddleTransposeUniformLaunch::new(
                    ScalarArg::new(*rows),
                    ScalarArg::new(*cols),
                    local_side_len,
                    ScalarArg::new(twiddles),
                );

                twiddle_transpose2d_tiled::launch_unchecked(
                    client,
                    CubeCount::Static(n_batches, n_row_tiles, n_col_tiles),
                    CubeDim::new_1d(*threads_per_cube),
                    input,
                    output,
                    ScalarArg::new(sign),
                    uniform,
                );
                std::mem::swap(&mut src, &mut dst);
            },
        }
    }

    if src != &array {
        std::mem::swap(array, tmp);
    }
}
