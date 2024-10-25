use rsplot1d::plot;
use std::f64::consts::PI;

fn rusanov_solution(nx: usize, t_max: f64, dt: f64, dx: f64) -> Vec<Vec<f64>> {
    // 空间区间 [-1, 2]
    let x_start = -1.0;
    let x_end = 2.0;
    let mut x: Vec<f64> = (0..=nx).map(|i| x_start + i as f64 * dx).collect();

    // 初始条件
    let mut u: Vec<f64> = x.iter().map(|&xi| {
        if xi < 0.0 {
            1.0
        } else if xi <= 1.0 {
            1.0 - xi
        } else {
            0.0
        }
    }).collect();

    let mut u_new = u.clone();
    let mut solutions = vec![u.clone()]; // 存储解

    let nt = (t_max / dt).ceil() as usize;
    let times = vec![0.5, 1.0, 2.0];
    let mut time_counter = 1;
    
    for n_step in 1..=nt {
        let current_time = n_step as f64 * dt;

        // Rusanov 模式数值更新
        for i in 1..nx {
            let f = |u: f64| 0.5 * u * u;
            let df = |u: f64| u;

            let a = df(u[i]).abs().max(df(u[i + 1]).abs());
            let flux_right = 0.5 * (f(u[i]) + f(u[i + 1])) - 0.5 * a * (u[i + 1] - u[i]);

            let a = df(u[i - 1]).abs().max(df(u[i]).abs());
            let flux_left = 0.5 * (f(u[i - 1]) + f(u[i])) - 0.5 * a * (u[i] - u[i - 1]);

            u_new[i] = u[i] - dt / dx * (flux_right - flux_left);
        }

        // 左边界条件
        u_new[0] = 1.0;
        // 右边界条件
        u_new[nx] = u_new[nx - 1];
        u = u_new.clone();

        // 存储在指定时间点的解
        if time_counter < times.len() && (current_time - times[time_counter]).abs() < 1e-5 {
            solutions.push(u.clone());
            time_counter += 1;
        }
    }

    solutions
}

fn main() {
    let nx_list = vec![50, 100, 1000, 10000];
    let t_max = 2.0;

    for &nx in &nx_list {
        let dx = 3.0 / nx as f64; // 空间步长
        let dt = 0.9 * dx; // 满足 CFL 条件
        let solutions = rusanov_solution(nx, t_max, dt, dx);

        // 生成空间网格
        let x_start = -1.0;
        let x_end = 2.0;
        let x: Vec<f64> = (0..=nx).map(|i| x_start + i as f64 * dx).collect();

        // 绘制数值解
        for (i, &time) in vec![0.5, 1.0, 2.0].iter().enumerate() {
            println!("绘制数值解：网格数量 N = {}, 时间 t = {}", nx, time);
            if let Err(err) = plot(&x, &solutions[i]) {
                eprintln!("绘图错误：{}", err);
            }
        }
    }

    println!("所有数值解绘图完成！");
}
