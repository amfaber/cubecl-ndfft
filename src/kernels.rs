use crate::complex::C32;
use cubecl::prelude::*;

// -------- small utilities --------

#[cube]
#[inline]
fn ceil_div(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

#[cube]
fn read_complex_from_shared(shared: &SharedMemory<f32>, idx_c: u32) -> C32 {
    let f = idx_c * 2;
    C32 {
        re: shared[f],
        im: shared[f + 1],
    }
}

#[cube]
fn write_complex_to_shared(shared: &mut SharedMemory<f32>, idx_c: u32, v: C32) {
    let f = idx_c * 2;
    shared[f] = v.re;
    shared[f + 1] = v.im;
}


#[derive(CubeType, CubeLaunch, Debug)]
pub struct TwiddleTransposeUniform {
    pub rows: u32, // in complex units
    pub cols: u32, // in complex units
    #[cube(comptime)]
    pub tile_side_len: u32, // in complex units
    pub apply_twiddles: u32, // 0 represents "dont apply twiddles"
}

/// Line length MUST be 2 for this kernel such that
/// we can load complex numbers as lines
#[cube(launch_unchecked)]
pub fn twiddle_transpose2d_tiled(
    input: &Array<Line<f32>>,
    output: &mut Array<Line<f32>>,
    sign: f32,
    uniform: TwiddleTransposeUniform,
) {
    // ---- thread/block ids ----
    let tid: u32 = UNIT_POS_X;
    let threads: u32 = CUBE_DIM_X;
    let batch: u32 = CUBE_POS_X;
    let tile_row: u32 = CUBE_POS_Y;
    let tile_col: u32 = CUBE_POS_Z;

    // ---- unpack uniforms ----
    let TwiddleTransposeUniform {
        rows,
        cols,
        tile_side_len,
        apply_twiddles,
    } = uniform;

    let n = rows * cols;
    let batch_stride = n;

    let start_row = tile_row * tile_side_len;
    let start_col = tile_col * tile_side_len;

    let base_idx = batch_stride * batch + start_row * cols + start_col;

    let tile_size = comptime!(tile_side_len * tile_side_len);
    let mut shared = SharedMemory::<f32>::new_lined(tile_size, 2u32);

    let mut flat_tile_id = tid;
    while flat_tile_id < tile_size {
        let row_in_tile = flat_tile_id / tile_side_len;
        let col_in_tile = flat_tile_id % tile_side_len;
        if (start_row + row_in_tile) < rows && (start_col + col_in_tile) < cols {
            let idx = base_idx + row_in_tile * cols + col_in_tile;

            let loaded = input[idx];

            shared[col_in_tile * tile_side_len + row_in_tile] = loaded;
        }
        flat_tile_id += threads;
    }

    sync_cube();

    let out_base = batch * batch_stride
             + start_col * rows      // transposed row origin
             + start_row; // transposed col origin

    let mut flat_tile_id = tid;
    while flat_tile_id < tile_size {
        let tile_row = flat_tile_id / tile_side_len; // 0..tile_side_len-1  (these are original cols)
        let tile_col = flat_tile_id % tile_side_len; // 0..tile_side_len-1  (these are original rows)
        let row_out = start_col + tile_row;
        let col_out = start_row + tile_col;

        if row_out < cols && col_out < rows {
            // read row-major from shared (already transposed on load)
            let mut value = shared[tile_row * tile_side_len + tile_col];
            if apply_twiddles != 0 {
                let c_value = C32 {
                    re: value[0],
                    im: value[1],
                };
                let twiddle = twiddle(sign, row_out * col_out, n);
                let corrected = c_value.mul(twiddle);
                value[0] = corrected.re;
                value[1] = corrected.im;
            }

            let out_idx = out_base + tile_row * rows + tile_col; // row stride is 'rows' in transposed
            output[out_idx] = value;
        }
        flat_tile_id += threads;
    }
}

#[cube]
fn twiddle(sign: f32, k: u32, n: u32) -> C32 {
    let theta = sign * 2.0 * std::f32::consts::PI * (k as f32) / n as f32;
    C32 {
        re: f32::cos(theta),
        im: f32::sin(theta),
    }
}

pub const R2: u32 = 2;

#[cube]
fn downgrade_const(x: u32) -> u32 {
    x
}

#[cube(launch_unchecked)]
pub fn fft1d_r2_fused(
    input: &mut Array<Line<f32>>, // AoS: [re, im, re, im, ...] (floats)
    sign: f32,
    #[comptime] fft_len: u32,
) {
    let r = 2;
    let thread_id: u32 = UNIT_POS_X;
    let num_threads: u32 = CUBE_DIM_X;
    let fft_idx: u32 = CUBE_POS_X;

    let vec_width = input.line_size();

    let base_scalar_c = fft_idx * fft_len;
    let base_f32 = base_scalar_c * 2;

    // Shared AoS ping-pong: two halves (each 2*T floats) → 4*T floats total
    let mut shared = SharedMemory::<f32>::new(fft_len * 4);

    // ---- Load: GLOBAL → SHARED[A] ----

    let fft_len_lines = ceil_div(fft_len * 2, vec_width);
    let lines_per_thread = ceil_div(fft_len_lines, num_threads);

    for j in 0..lines_per_thread {
        let line_idx = thread_id + j * num_threads;
        if line_idx < fft_len_lines {
            let global_line_idx = (base_f32 / vec_width) + line_idx;
            let line_vals = input[global_line_idx];
            for lane in 0..vec_width {
                let s = line_idx * vec_width + lane;
                shared[s] = line_vals[lane];
            }
        }
    }
    sync_cube();

    // ---- Fused stages in shared (A <-> B), Stockham autosort every stage ----
    let mut src_base_c = 0; // complex index base for src half
    let mut dst_base_c = downgrade_const(fft_len); // complex index base for dst half

    let mut ns = 1;
    let n = fft_len;
    let butterflies = fft_len / r;

    while ns < n {
        let mut butterfly_idx: u32 = thread_id;
        while butterfly_idx < butterflies {
            let k = butterfly_idx % ns;
            let in_index0 = butterfly_idx;
            let in_index1 = butterfly_idx + butterflies;

            let c0 = read_complex_from_shared(&shared, src_base_c + in_index0);
            let c1 = read_complex_from_shared(&shared, src_base_c + in_index1);

            let w = twiddle(sign, k, ns * r);
            let t1 = c1.mul(w);

            let o0 = c0.add(t1);
            let o1 = c0.sub(t1);

            let out_index0 = (butterfly_idx / ns) * ns * r + k;
            let out_index1 = out_index0 + ns;

            write_complex_to_shared(&mut shared, dst_base_c + out_index0, o0);
            write_complex_to_shared(&mut shared, dst_base_c + out_index1, o1);

            butterfly_idx += num_threads;
        }
        sync_cube();

        ns *= R2;

        let tmp = src_base_c;
        src_base_c = dst_base_c;
        dst_base_c = tmp;
    }

    // ---- Store: SHARED[src] → GLOBAL ----
    let final_base_f32 = src_base_c * 2;
    for j in 0..lines_per_thread {
        let line_idx = thread_id + j * num_threads;
        if line_idx < fft_len_lines {
            let global_line_idx = (base_f32 / vec_width) + line_idx;
            let mut out_line = Line::<f32>::empty(vec_width);
            let s = line_idx * vec_width;
            for lane in 0..vec_width {
                out_line[lane] = shared[final_base_f32 + s + lane];
            }
            input[global_line_idx] = out_line;
        }
    }
}
