#[allow(dead_code)]
pub fn run() {
    use rsplot1d::plot;
    // 空间离散点数量的数组，用于网格细化
    let space_points_list = vec![50, 100, 200, 400, 800];
    // 保存不同网格下的误差范数
    let mut l1_errors = Vec::new();
    let mut l2_errors = Vec::new();
    let mut dx_values = Vec::new();

    // 循环不同的网格尺寸
    for &num_space_points in &space_points_list {
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
                    // 处理右边界（此处假设为零）
                    u_next[i] = 0.0;
                }
            }

            // 更新当前时间步的解
            u_current.copy_from_slice(&u_next);
        }

        // 计算精确解
        let exact_solution: Vec<f64> = x_values
            .iter()
            .map(|&x| {
                if x <= c * t_max {
                    (- (t_max - x / c)).exp()
                } else {
                    0.0
                }
            })
            .collect();

        // 计算误差
        let mut l1_error = 0.0;
        let mut l2_error = 0.0;

        for i in 0..=num_space_points {
            let error = (u_current[i] - exact_solution[i]).abs();
            l1_error += error * dx;
            l2_error += error * error * dx;
        }

        l2_error = l2_error.sqrt();

        // 保存误差和网格尺寸
        l1_errors.push(l1_error);
        l2_errors.push(l2_error);
        dx_values.push(dx);
    }

    // 输出误差结果
    println!("dx\tL1 Error\tL2 Error");
    for i in 0..dx_values.len() {
        println!(
            "{:.5}\t{:.5e}\t{:.5e}",
            dx_values[i], l1_errors[i], l2_errors[i]
        );
    }

    // 使用 rsplot1d::plot 绘制误差随网格尺寸变化的图像
    // 计算 ln(Δx) 和 ln(误差)
    let ln_dx_values: Vec<f64> = dx_values.iter().map(|&dx| dx.ln()).collect();
    let ln_l1_errors: Vec<f64> = l1_errors.iter().map(|&err| err.ln()).collect();
    let ln_l2_errors: Vec<f64> = l2_errors.iter().map(|&err| err.ln()).collect();

    // 绘制 ln(L1 Error) 对 ln(Δx) 的图像
    // plot(&ln_dx_values, &ln_l1_errors, &vec![]);

    // 如果需要同时绘制 L1 和 L2 误差，可以修改 plot 函数以支持多个曲线
    // 假设 rsplot1d::plot 支持同时绘制多条曲线，可以传入多个数据集
    plot(&ln_dx_values, &ln_l1_errors, &ln_l2_errors);

    // 如果 rsplot1d::plot 只支持两组数据，我们可以分别绘制 L1 和 L2 误差
    // 绘制 ln(L2 Error) 对 ln(Δx) 的图像
    // plot(&ln_dx_values, &ln_l2_errors, &vec![]);
}
