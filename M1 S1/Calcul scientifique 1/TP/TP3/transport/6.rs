#[allow(dead_code)]
pub fn run() {
    use rsplot1d::plot;
    // 空间离散点数量
    let num_space_points = 100;
    // 空间步长 Δx
    let dx = 1.0 / num_space_points as f64;
    // 速度常数 c
    let c = 1.0;

    // 最大时间 T
    let t_max = 2.0; // 增大 t_max
    // 时间步长 Δt
    let dt = 0.5 * dx / c; // 这里的 dt 选择较小，但中心差分格式无条件不稳定

    // ν = c * Δt / (2Δx)
    let nu = c * dt / (2.0 * dx);

    // 初始条件：u(x, 0) = 0
    let mut u_current = vec![0.0; num_space_points + 2]; // 加 2 以处理边界
    let mut u_next = u_current.clone();

    // 空间坐标数组 x
    let x_values: Vec<f64> = (0..=num_space_points + 1)
        .map(|i| (i as f64 - 1.0) * dx)
        .collect();

    // 用于保存不同时间的解
    let mut snapshots = Vec::new();
    let snapshot_times = vec![0.5, 1.0, 1.5, 2.0];

    // 时间变量初始化
    let mut time = 0.0;

    // 快照时间索引
    let mut snapshot_index = 0;

    // 时间循环
    while time < t_max {
        // 更新时间
        time += dt;

        // 边界条件：u(0, t) = e^{-t}
        u_current[1] = (-time).exp(); // 对应于 x = 0 的位置

        // 更新内部空间点的值（中心差分格式）
        for i in 1..=num_space_points {
            u_next[i] = u_current[i] - nu * (u_current[i + 1] - u_current[i - 1]);
        }

        // 更新边界条件（简单处理）
        u_next[0] = u_next[1];
        u_next[num_space_points + 1] = u_next[num_space_points];

        // 更新当前时间步的解
        u_current.copy_from_slice(&u_next);

        // 保存指定时间的解（考虑浮点数精度）
        let epsilon = 1e-6;
        if snapshot_index < snapshot_times.len()
            && (time - snapshot_times[snapshot_index]).abs() < epsilon
        {
            snapshots.push((time, u_current.clone()));
            snapshot_index += 1;
        }

        // 为了更明显地观察不稳定性，可以在每个时间步都保存快照
        // snapshots.push((time, u_current.clone()));
    }

    // 绘制不同时间的解
    for (time, u) in snapshots {
        let u_numerical = &u[1..=num_space_points + 1];
        let x_plot = &x_values[1..=num_space_points + 1];
        println!("Time = {}", time);
        plot(x_plot, u_numerical, u_numerical);
    }
}
