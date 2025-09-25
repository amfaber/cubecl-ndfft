use cubecl::{
    CubeCount, CubeDim, Runtime, client::ComputeClient, prelude::ScalarArg, server::Handle,
};

use crate::kernels::{
    TwiddleTransposeUniformLaunch, fft1d_r2_fused, fft1d_r2_naive, twiddle_transpose2d_tiled,
};

#[derive(Debug)]
pub struct TransposeOp {
    pub rows: u32,
    pub cols: u32,
    pub twiddles: bool,
}

#[derive(Debug)]
pub enum Op {
    Fft { len: u32 },
    FftNaive { len: u32, ns: u32 },
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

pub enum KernelKind {
    Local,
    Global,
}

pub fn plan_fft(
    dimensions: Vec<usize>,
    has_batch: bool,
    fuse_size: u32,
    kernel_kind: KernelKind,
) -> FftPlan {
    let mut ops = vec![];
    let buffer_len = dimensions.iter().product::<usize>();

    for (index, &dimension) in dimensions.iter().enumerate().rev() {
        if !(index == 0 && has_batch) {
            match kernel_kind {
                KernelKind::Local => {
                    plan1d(dimension as u32, fuse_size, &mut ops);
                }
                KernelKind::Global => {
                    plan1d_naive(dimension as u32, &mut ops);
                }
            }
        }
        let rows = (buffer_len / dimension) as u32;
        if rows != 1 && dimension != 1 {
            ops.push(Op::Transpose(TransposeOp {
                rows,
                cols: dimension as u32,
                twiddles: false,
            }))
        }
    }
    FftPlan { ops, buffer_len }
}

fn plan1d_naive(fft_len: u32, ops: &mut Vec<Op>) {
    let mut ns = 1;
    while ns < fft_len {
        ops.push(Op::FftNaive {
            len: fft_len,
            ns: ns,
        });
        ns *= 2;
    }
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
    mut array: Handle,
    tmp: &mut Handle,
    plan: &FftPlan,
    direction: Direction,
    client: &ComputeClient<RT::Server, RT::Channel>,
    params: &FftParams,
) -> Handle {
    let sign = match direction {
        Direction::Forward => -1.,
        Direction::Inverse => 1.,
    };
    let mut src = &(&mut array);
    let pointer = src as *const _;
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
                    *buffer_len * 2,
                    *vec_width,
                );

                fft1d_r2_fused::launch_unchecked(
                    client,
                    CubeCount::new_1d((buffer_len * 2 / *len as usize) as u32),
                    CubeDim::new_1d(*threads_per_cube),
                    input,
                    ScalarArg::new(sign),
                    *len,
                );
            },
            Op::FftNaive { len, ns } => unsafe {
                let input =
                    cubecl::prelude::ArrayArg::<RT>::from_raw_parts::<f32>(src, *buffer_len * 2, 2);

                let output =
                    cubecl::prelude::ArrayArg::<RT>::from_raw_parts::<f32>(dst, *buffer_len * 2, 2);

                let butterflies = (buffer_len / 2) as u32;
                let naive_dispatcher =
                    dispatcher_flat(butterflies as u64, *threads_per_cube as u64);

                fft1d_r2_naive::launch_unchecked::<RT>(
                    client,
                    naive_dispatcher,
                    CubeDim::new_1d(*threads_per_cube),
                    input,
                    output,
                    ScalarArg::new(sign),
                    ScalarArg::new(*len),
                    ScalarArg::new(*ns),
                );
                std::mem::swap(&mut src, &mut dst);
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
                    cubecl::prelude::ArrayArg::<RT>::from_raw_parts::<f32>(src, *buffer_len * 2, 2);

                let output =
                    cubecl::prelude::ArrayArg::<RT>::from_raw_parts::<f32>(dst, *buffer_len * 2, 2);

                let twiddles = *twiddles as u32;
                let uniform = TwiddleTransposeUniformLaunch::<RT>::new(
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

    if (src as *const _) != pointer {
        std::mem::swap(&mut array, tmp);
    }
    array
}

/// For dispatching on a grid to ensure at least "len" invocations.
/// No guarantuees are made about the layout of the workgrid. Tries to minimize
/// invocations above "len". The shader should calculate its own flat index to
/// ensure consitency.
///
/// Useful for large dispatches that can't fit into a single dimension due to
/// hardware limits.
pub fn dispatcher_flat(len: u64, wg_size: u64) -> CubeCount {
    let n_workgroups = len as f64 / wg_size as f64;
    let exact_workgroups = (len as usize + wg_size as usize - 1) / wg_size as usize;
    if exact_workgroups < (1 << 16) {
        return CubeCount::new_1d(exact_workgroups as u32);
    }
    let primes = slow_primes::Primes::sieve((n_workgroups.sqrt() * 1.1) as usize);
    let mut work_group = [1, 1, 1];
    let mut keep_going = true;
    let mut offset = 0;
    while keep_going {
        let factors = primes.factor(exact_workgroups + offset).unwrap();
        let mut idx = 0;
        let mut iter = factors.into_iter();
        keep_going = loop {
            let Some((base, exp)) = iter.next() else {
                break false;
            };
            if base > (1 << 16) {
                offset += 1;
                break true;
            }
            for _ in 0..exp {
                if work_group[idx] * base < (1 << 16) {
                    work_group[idx] *= base;
                } else {
                    idx += 1;
                    work_group[idx] *= base;
                }
            }
        }
    }
    CubeCount::Static(
        work_group[0] as u32,
        work_group[1] as u32,
        work_group[2] as u32,
    )
}
