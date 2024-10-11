#[allow(dead_code)]
pub fn run() {
    use rsplot1d::plot;
    // 空间离散点数量
    let num_space_points = 200;
    // 空间步长 Δx
    let dx = 1.0 / num_space_points as f64;
    // 速度常数 c
    let c = 1.0;

    // 最大时间 T
    let t_max = 0.5;
    // CFL 数
    let lambda = 0.9; // 接近稳定性边界
    // 时间步长 Δt
    let dt = lambda * dx / c;

    // 初始条件：u(x, 0)
    let mut u_current = vec![0.0; num_space_points + 1];
    let mut u_next = u_current.clone();

    let x_values: Vec<f64> = (0..=num_space_points)
        .map(|i| i as f64 * dx)
        .collect();

    // 设置初始条件（间断）
    for i in 0..=num_space_points {
        if x_values[i] < 0.5 {
            u_current[i] = 1.0;
        } else {
            u_current[i] = 0.0;
        }
    }

    // 时间变量初始化
    let mut time = 0.0;

    // 时间循环
    while time < t_max {
        time += dt;

        // 周期性边界条件
        u_current[0] = u_current[num_space_points - 1];
        u_current[num_space_points] = u_current[1];

        // 更新内部空间点的值
        for i in 1..num_space_points {
            let u_ip1 = u_current[i + 1];
            let u_i = u_current[i];
            let u_im1 = u_current[i - 1];

            u_next[i] = u_i
                - 0.5 * lambda * (u_ip1 - u_im1)
                + 0.5 * lambda * lambda * (u_ip1 - 2.0 * u_i + u_im1);
        }

        // 更新边界条件
        u_next[0] = u_next[num_space_points - 1];
        u_next[num_space_points] = u_next[1];

        // 更新当前时间步的解
        u_current.copy_from_slice(&u_next);

        // 计算 L2 和 L∞ 范数
        let mut l2_norm = 0.0;
        let mut max_u = f64::MIN;
        let mut min_u = f64::MAX;
        for i in 0..=num_space_points {
            let ui = u_current[i];
            l2_norm += ui * ui * dx;
            if ui > max_u {
                max_u = ui;
            }
            if ui < min_u {
                min_u = ui;
            }
        }
        l2_norm = l2_norm.sqrt();
        println!(
            "Time = {:.3}, L2 Norm = {:.5}, Max u = {:.5}, Min u = {:.5}",
            time, l2_norm, max_u, min_u
        );
    }

    // 绘制结果
    plot(&x_values, &u_current, &u_current);
}
