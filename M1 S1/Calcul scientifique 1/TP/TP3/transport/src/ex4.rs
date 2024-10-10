#[allow(dead_code)]
pub fn run() {
    use rsplot1d::plot;
    // 空间离散点的数量
    let num_space_points = 100;
    // 空间步长 Δx
    let dx = 1.0 / num_space_points as f64;
    // 速度常数 c
    let c = 1.0;

    // 最大时间 T
    let t_max = 0.5;
    // CFL 数值（满足 0 < cfl_number <= 1）
    let cfl_number = 0.8;
    // 时间步长 Δt，根据 CFL 条件计算
    let dt = cfl_number * dx / c;
    // v = c * Δt / Δx，实际上等于 CFL 数值
    let v = c * dt / dx;

    // 初始条件：u(x, 0) = 0
    let mut u_current = vec![0.0; num_space_points + 1];
    // 下一时间步的解
    let mut u_next = u_current.clone();

    // 空间坐标数组 x
    let x_values: Vec<f64> = (0..=num_space_points)
        .map(|i| i as f64 * dx)
        .collect();

    // 时间变量初始化
    let mut time = 0.0;

    // 时间循环
    while time < t_max {
        // 更新时间
        time += dt;
        // 边界条件：u(0, t) = e^{-t}
        u_next[0] = (-time).exp();

        // 更新内部空间点的值
        for i in 1..=num_space_points {
            if i < num_space_points {
                u_next[i] = (1.0 - v) * u_current[i] + v * u_current[i - 1];
            } else {
                // 处理右边界（此处假设为零导数边界条件）
                u_next[i] = u_current[i];
            }
        }

        // 更新当前时间步的解
        for i in 0..=num_space_points {
            u_current[i] = u_next[i];
        }
    }

    // 绘制结果
    plot(&x_values, &u_current, &u_next);
}
