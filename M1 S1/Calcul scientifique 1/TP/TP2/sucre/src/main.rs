use skyrs::Sky;
use rsplot1d::plot1d;

fn main() {
    // 空间和时间参数
    let l = 1.0; // 空间长度 L
    let nx = 100; // 空间离散点数
    let dx = l / nx as f64; // 空间步长 Δx

    let dt = 0.00001; // 时间步长 Δt
    let t_max = 0.1; // 最大时间
    let nt = (t_max / dt) as usize; // 时间步数

    let theta = 0.5; // θ-格式参数，0<=theta<=1

    // 计算 α = Δt / Δx^2
    let alpha = dt / dx.powi(2);

    // 初始化空间网格
    let x: Vec<f64> = (0..=nx).map(|i| i as f64 * dx).collect();

    // 初始条件 u(x, 0)
    let mut u: Vec<f64> = x
        .iter()
        .map(|&xi| {
            if xi >= 0.375 && xi <= 0.625 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    // 构建稀疏矩阵 A
    let mut coo = Vec::new();

    // 构建矩阵的对角线元素和上下对角线元素
    let a = 1.0 + 2.0 * theta * alpha;
    let b = -theta * alpha;

    for i in 0..=nx {
        if i == 0 {
            // 左边界点，处理 Neumann 边界条件
            coo.push((i, i, a + b)); // 合并虚拟点
            coo.push((i, i + 1, b + b));
        } else if i == nx {
            // 右边界点，处理 Neumann 边界条件
            coo.push((i, i - 1, b + b));
            coo.push((i, i, a + b));
        } else {
            // 内部点
            coo.push((i, i - 1, b));
            coo.push((i, i, a));
            coo.push((i, i + 1, b));
        }
    }

    // 构建 Sky 对象
    let mut sky = Sky::new(coo);

    // 时间迭代
    for n in 1..=nt {
        let t = n as f64 * dt;

        // 构建右端项 b
        let mut b_vec = vec![0.0; nx + 1];

        for i in 0..=nx {
            if i == 0 {
                // 左边界点
                b_vec[i] = (1.0 - 2.0 * (1.0 - theta) * alpha) * u[i]
                    + 2.0 * (1.0 - theta) * alpha * u[i + 1];
            } else if i == nx {
                // 右边界点
                b_vec[i] = (1.0 - 2.0 * (1.0 - theta) * alpha) * u[i]
                    + 2.0 * (1.0 - theta) * alpha * u[i - 1];
            } else {
                // 内部点
                b_vec[i] = (1.0 - 2.0 * (1.0 - theta) * alpha) * u[i]
                    + (1.0 - theta) * alpha * (u[i - 1] + u[i + 1]);
            }
        }

        // 求解线性方程组 A u_new = b
        let u_new = sky.solve(b_vec).expect("求解失败");

        // 更新 u
        u = u_new;

        // 绘制结果（每隔一定的时间步绘制一次，例如每 1000 个时间步）
        if n % 500 == 0 {
            println!("绘制时间 t = {:.5}", t);
            plot1d(&x, &u, &u);
        }
    }
    println!("计算完成。");
}