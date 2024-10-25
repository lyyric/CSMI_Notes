use rsplot1d::plot;
use std::f64;

fn exact_solution(x: f64, t: f64) -> f64 {
    if t < 1.0 {
        if x <= t {
            1.0
        } else if x > t && x < 1.0 {
            (1.0 - x) / (1.0 - t)
        } else {
            0.0
        }
    } else {
        let shock_position = 0.5 * (t + 1.0);
        if x < shock_position {
            1.0
        } else {
            0.0
        }
    }
}

fn main() {
    let nx = 1000;
    let x_start = -1.0;
    let x_end = 2.0;
    let dx = (x_end - x_start) / nx as f64;
    let x: Vec<f64> = (0..=nx).map(|i| x_start + i as f64 * dx).collect();

    let times = vec![0.5, 1.0, 2.0];

    for &t in &times {
        let u: Vec<f64> = x.iter().map(|&xi| exact_solution(xi, t)).collect();

        // 绘制精确解
        println!("t = {}", t);
        plot(&x, &u, &u)
    }
}
