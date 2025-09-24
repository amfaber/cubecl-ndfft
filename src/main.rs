#![allow(unused)]
use bytemuck::Zeroable;
use cubecl::benchmark::Benchmark;
use cubecl::future::{self, block_on};
use cubecl::server::Handle;
use cubecl_ndfft::plan::{Direction, FftParams, FftPlan, execute_ops, plan_fft};
use ndarray::{Array2, ArrayViewMut2};
use ndrustfft::FftHandler;
use num_complex::{Complex, Complex32};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use rustfft::{FftDirection, FftPlanner};

use cubecl::prelude::*;
use cubecl_ndfft::kernels::{
    TwiddleTransposeUniform, TwiddleTransposeUniformLaunch, fft1d_r2_fused,
    twiddle_transpose2d_tiled,
};

fn aos_to_complex(v: &[f32]) -> Vec<Complex32> {
    v.chunks_exact(2)
        .map(|c| Complex32::new(c[0], c[1]))
        .collect()
}

fn complex_to_aos(v: &[Complex32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * v.len());
    for z in v {
        out.push(z.re);
        out.push(z.im);
    }
    out
}

fn rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    (acc / (a.len() as f32)).sqrt()
}

fn fft_2d_cpu(array: &mut Array2<Complex32>, tmp: &mut Array2<Complex32>) {
    let (rows, cols) = array.dim();
    let now = std::time::Instant::now();
    let row_handler = FftHandler::new(rows);
    let col_handler = FftHandler::new(cols);
    dbg!(now.elapsed());
    // let mut tmp = Array2::zeros(array.dim());
    
    let now = std::time::Instant::now();
    ndrustfft::ndfft(&array, tmp, &col_handler, 1);
    ndrustfft::ndfft(&tmp, array, &row_handler, 0);
    dbg!(now.elapsed());
    // ndrustfft::ndfft(&array, &mut tmp, &row_handler, 0);
    // ndrustfft::ndfft(&tmp, array, &col_handler, 1);
}

fn main() {
    // Problem
    let rows = 512;
    let cols = 512;
    let n = rows * cols;
    let duplicates = 100;
    dbg!(n);
    // let vec_width: u32 = 4;
    // let threads: u32 = 256;

    // Host input AoS
    // let mut rng = StdRng::seed_from_u64(123);
    // let host_in = (0..(2 * n)).map(|x| (x % 4) as f32).collect::<Vec<_>>();

    let mut in_array = Array2::from_shape_fn((rows, cols as usize), |(i, j)| {
        Complex32 {
            re: i as f32,
            im: j as f32,
        }
        // PrettierIndex([i as f32, j as f32])
    });
    // dbg!(&in_array);
    // let host_in = bytemuck::cast_slice::<PrettierIndex, f32>(in_array.as_slice().unwrap()).to_vec();

    let host_in = (0..duplicates)
        .map(|_| in_array.clone())
        .collect::<Vec<_>>();

    // let host_in = (0..(2 * n))
    //     .map(|x| x as f32)
    //     // .map(|x| rand::random_range(0f32..1f32))
    //     // .map(|x| (x % 4) as f32)
    //     .collect::<Vec<_>>();

    // CPU reference (rustfft)
    

    let mut tmp = Array2::zeros(in_array.dim());
    // let now = std::time::Instant::now();

    fft_2d_cpu(&mut in_array, &mut tmp);
    // let cpu_out = host_in
    //     .clone()
    //     .into_iter()
    //     .flat_map(|mut array| {
    //         fft_2d_cpu(&mut array);
    //         array
    //     })
    //     .collect::<Vec<_>>();
    // let cpu_out = host_in
    //     .clone()
    //     .into_iter()
    //     .flat_map(|mut array| {
    //         fft_2d_cpu(&mut array);
    //         array
    //     })
    //     .collect::<Vec<_>>();
    // let cpu_out = complex_to_aos(&cpu_out);
    // let mut cpu = aos_to_complex(&host_in);
    // fft.process(&mut cpu);
    // let cpu_out = complex_to_aos(&cpu);

    // dbg!(&cpu_out);

    let host_in = host_in
        .into_iter()
        .map(|arr| arr.into_iter().flat_map(|c| [c.re, c.im]))
        .flatten()
        .collect::<Vec<_>>();
    // cpu_six_step(&in_array);

    // Twiddles for fused (T=N)
    // let (tw, _stages) = make_fused_twiddles_r2(1, n);
    // dbg!(_stages);
    // dbg!(&tw);

    // --- GPU run (fully fused) ---
    //
    // type RT = cubecl::cuda::CudaRuntime; // adjust runtime if needed
    type RT = cubecl::wgpu::WgpuRuntime; // adjust runtime if needed
    let device = <RT as Runtime>::Device::default();
    let client = RT::client(&device);

    let plan = plan_fft(vec![duplicates, rows, cols], true, dbg!(1 << 12));

    dbg!(&plan);

    // let n_elem = n;
    // let mut input = client.create(f32::as_bytes(&host_in));
    // let mut tmp = client.empty(host_in.len() * std::mem::size_of::<f32>());
    // let parameters = FftParams {
    //     vec_width: 4,
    //     local_side_len: 64,
    //     threads_per_cube: 256,
    // };

    // execute_ops::<RT>(
    //     &mut input,
    //     &mut tmp,
    //     &plan,
    //     cubecl_ndfft::plan::Direction::Forward,
    //     &client,
    //     &parameters,
    // );

    // let gpu_out = bytemuck::cast_slice::<u8, f32>(&client.read_one(input)).to_vec();

    // dbg!(&gpu_out);
    // dbg!(rmse(&cpu_out, &gpu_out));
    // let out = bytemuck::cast_slice::<u8, PrettierIndex>(&client.read_one(input)).to_vec();

    // let mut out = Array2::from_shape_vec((2, 4), out)
    //     .unwrap()
    //     .mapv(|x| Complex32 {
    //         re: x.0[0],
    //         im: x.0[1],
    //     });
    // print_src("gpu_out", &out.view_mut());

    // pure_transpose_test::<RT>(&client);

    // cpu_four_step(&host_in);

    // let full_data = (0..duplicates)
    //     .flat_map(|_| host_in.clone())
    //     .collect::<Vec<_>>();

    // let gpu_out = correctness_tests::<RT>(&client, full_data, n, 4);

    // dbg!(gpu_out);

    let mut bench = FftBenchmark::<RT> {
        client: client.clone(),
        full_data: host_in,
        plan,
    };

    // bench.vec_width = 4;
    let timing_method = cubecl::benchmark::TimingMethod::Device;
    let durations = bench.run(timing_method).unwrap();
    dbg!(durations);
    // bench.vec_width = 1;
    // let durations = bench.run(timing_method).unwrap();
    // dbg!(durations);
    // let twiddle_handle = client.create(f32::as_bytes(&tw));
    // Upload
}

fn cpu_six_step(array: &Array2<PrettierIndex>) {
    let (t, r) = array.dim();
    let n = t * r;
    dbg!(t);
    let first_fft = ndrustfft::FftHandler::new(t);
    let second_fft = ndrustfft::FftHandler::new(r);
    let mut complex = array.mapv(|v| Complex32 {
        re: v.0[0],
        im: v.0[1],
    });
    let mut tmp = Array2::zeros(array.dim());

    let mut src = complex.view_mut();
    let mut dst = tmp.view_mut();

    // print_src("six step start", &src);

    src = src.reversed_axes();
    dst = dst.reversed_axes();
    dst.fill(Complex32::ZERO);
    // print_src("six step transpose", &src);

    ndrustfft::ndfft(&src, &mut dst, &first_fft, 1);
    std::mem::swap(&mut src, &mut dst);

    // print_src("six step first fft", &src);

    src = src.reversed_axes();
    dst = dst.reversed_axes();
    dst.fill(Complex32::ZERO);
    ndarray::Zip::indexed(&mut src).for_each(|(i, j), val| {
        let twiddle = Complex32::from_polar(1., -std::f32::consts::TAU * (i * j) as f32 / n as f32);
        *val = *val * twiddle
    });
    // print_src("six step twiddle transpose", &src);
    ndrustfft::ndfft(&src, &mut dst, &second_fft, 1);
    std::mem::swap(&mut src, &mut dst);
    // dbg!(&src);
    // print_src("six step second fft", &src);

    src = src.reversed_axes();
    dst = dst.reversed_axes();
    dst.fill(Complex32::ZERO);

    print_src("six step final transpose", &src);

    let src = src.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<_>>();

    // dbg!(&src);
}

fn print_src(message: &str, src: &ArrayViewMut2<Complex32>) {
    let prettier = unsafe {
        std::mem::transmute::<&ArrayViewMut2<Complex32>, &ArrayViewMut2<PrettierIndex>>(src)
    };
    println!("{message}\n{:#?}", prettier);
}

#[derive(bytemuck::Pod, Clone, Copy, Zeroable)]
#[repr(transparent)]
struct PrettierIndex([f32; 2]);
impl std::fmt::Debug for PrettierIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}, {})", self.0[0], self.0[1]))
    }
}

fn pure_transpose_test<RT: Runtime>(client: &ComputeClient<RT::Server, RT::Channel>) {
    let rows = 20;
    let cols = 20;
    let data =
        ndarray::Array::from_shape_fn([rows, cols], |(i, j)| PrettierIndex([i as f32, j as f32]));

    dbg!(&data);
    let bytes_data = bytemuck::cast_slice::<_, u8>(data.as_slice().unwrap());
    let in_handle = client.create(bytes_data);
    let out_handle = client.empty(bytes_data.len());

    let n_batches = 1;
    let n_tile_rows = 40;
    let n_tile_cols = 40;

    let input_arg = unsafe { ArrayArg::<RT>::from_raw_parts::<f32>(&in_handle, rows * cols, 2) };
    let output_arg = unsafe { ArrayArg::<RT>::from_raw_parts::<f32>(&out_handle, rows * cols, 2) };

    let tile_side_len = 4;
    let uniform = TwiddleTransposeUniformLaunch::new(
        ScalarArg::new(rows as u32),
        ScalarArg::new(cols as u32),
        &tile_side_len,
        ScalarArg::new(1),
    );

    unsafe {
        twiddle_transpose2d_tiled::launch_unchecked(
            client,
            CubeCount::Static(n_batches, n_tile_rows, n_tile_cols),
            CubeDim::new(256, 1, 1),
            input_arg,
            output_arg,
            ScalarArg::new(-1.0),
            uniform,
        );
    }

    let out = bytemuck::cast_slice::<u8, PrettierIndex>(&client.read_one(out_handle)).to_vec();

    let recovered = ndarray::Array::from_shape_vec([cols, rows], out).unwrap();
    dbg!(&recovered);
}

fn correctness_tests<RT: Runtime>(
    client: &ComputeClient<RT::Server, RT::Channel>,
    full_data: Vec<f32>,
    n: u32,
    t: u32,
) -> Vec<f32> {
    let vec_width = 2;
    let n_elem = full_data.len() as u32 / 2;
    let handle = client.create(f32::as_bytes(&full_data));
    let output_handle = client.empty(full_data.len() * std::mem::size_of::<f32>());
    let d_input =
        unsafe { ArrayArg::<RT>::from_raw_parts::<f32>(&handle, full_data.len(), vec_width as u8) };
    let r = n / t;
    let n_batches = n_elem as u32 / n;
    let n_ffts = n_elem as u32 / t;
    unsafe {
        fft1d_r2_fused::launch_unchecked::<RT>(
            &client,
            CubeCount::Static(n_ffts, 1, 1),
            CubeDim::new(256, 1, 1),
            d_input,
            ScalarArg::new(-1.),
            t,
        );
    }

    let n_cols = t;
    let n_rows = r;

    let tile_side_len = 64;

    let n_tile_rows = n_rows.div_ceil(tile_side_len);
    let n_tile_cols = n_cols.div_ceil(tile_side_len);

    let input_arg = unsafe { ArrayArg::<RT>::from_raw_parts::<f32>(&handle, full_data.len(), 2) };
    let output_arg =
        unsafe { ArrayArg::<RT>::from_raw_parts::<f32>(&output_handle, full_data.len(), 2) };

    let uniform = TwiddleTransposeUniformLaunch::new(
        ScalarArg::new(n_rows),
        ScalarArg::new(n_cols),
        &tile_side_len,
        ScalarArg::new(1),
    );

    unsafe {
        twiddle_transpose2d_tiled::launch_unchecked(
            client,
            CubeCount::Static(n_batches, n_tile_rows, n_tile_cols),
            CubeDim::new(256, 1, 1),
            input_arg,
            output_arg,
            ScalarArg::new(-1.0),
            uniform,
        );
    }

    let n_ffts = n_elem / r;
    let output_arg = unsafe {
        ArrayArg::<RT>::from_raw_parts::<f32>(&output_handle, full_data.len(), vec_width as u8)
    };
    unsafe {
        fft1d_r2_fused::launch_unchecked::<RT>(
            &client,
            CubeCount::Static(n_ffts, 1, 1),
            CubeDim::new(256, 1, 1),
            output_arg,
            ScalarArg::new(-1.),
            r,
        );
    }

    let output = f32::from_bytes(&client.read_one(output_handle)).to_vec();

    // let input_arg = unsafe { ArrayArg::<RT>::from_raw_parts::<f32>(&handle, full_data.len(), 2) };
    // let output_arg =
    //     unsafe { ArrayArg::<RT>::from_raw_parts::<f32>(&output_handle, full_data.len(), 2) };

    // let uniform = TwiddleTransposeUniformLaunch::new(
    //     ScalarArg::new(n_cols),
    //     ScalarArg::new(n_rows),
    //     &tile_side_len,
    //     ScalarArg::new(0),
    // );

    // unsafe {
    //     twiddle_transpose2d_tiled::launch_unchecked(
    //         client,
    //         CubeCount::Static(n_batches, n_tile_cols, n_tile_rows),
    //         CubeDim::new(256, 1, 1),
    //         output_arg,
    //         input_arg,
    //         ScalarArg::new(-1.0),
    //         uniform,
    //     );
    // }

    // let output = f32::from_bytes(&client.read_one(handle)).to_vec();

    output
}

struct FftBenchmark<R: Runtime> {
    client: ComputeClient<R::Server, R::Channel>,
    plan: FftPlan,
    full_data: Vec<f32>,
}

impl<R: Runtime> Benchmark for FftBenchmark<R> {
    type Input = (Handle, Handle);

    type Output = Handle;

    fn prepare(&self) -> Self::Input {
        let input = self.client.create(f32::as_bytes(&self.full_data));
        let tmp = self.client.create(f32::as_bytes(&self.full_data));
        (input, tmp)
    }

    fn execute(&self, input: Self::Input) -> Result<Self::Output, String> {
        let (mut input, mut tmp) = input;
        execute_ops::<R>(
            &mut input,
            &mut tmp,
            &self.plan,
            Direction::Forward,
            &self.client,
            &FftParams {
                vec_width: 4,
                local_side_len: 64,
                threads_per_cube: 256,
            },
        );

        Ok(input)
    }

    fn name(&self) -> String {
        "Fused fft kernel in local memory".to_string()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }
}
