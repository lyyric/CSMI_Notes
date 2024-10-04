use rsplot1d::plot;

fn main() {
    let c = 1.0;
    let l = 1.0;
    let t = 1.0;
    let nx = 100;
    let dx = l / nx as f64;

    let lambda = 0.5;
    let dt = lambda * dx / c;
    let nt = (t / dt).ceil() as usize;
    let dt = t / nt as f64;
    let lambda = c * dt / dx;

    println!("CFL lambda = {}", lambda);

    let mut u = vec![vec![0.0; nx + 1]; nt + 1];
    let x = (0..=nx).map(|i| i as f64 * dx).collect::<Vec<f64>>();

    for n in 0..nt {
        let t = n as f64 * dt;
        u[n][0] = (-t).exp();

        for i in 1..=nx {
            u[n + 1][i] = (1. - lambda) * u[n][i] + lambda * u[n][i - 1];
        }
    }

    let t_final = nt as f64 * dt;
    u[nt][0] = (-t_final).exp();

    let final_u = &u[nt];
    plot(&x, final_u, final_u);
}
