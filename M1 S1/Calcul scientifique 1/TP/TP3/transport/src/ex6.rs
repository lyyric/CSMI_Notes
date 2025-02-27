#[allow(dead_code)]
pub fn run() {
    use rsplot1d::plot;

    let num_space_points = 100;
    let dx = 1.0 / num_space_points as f64;
    let c = 1.0;

    let t_max = 2.0;
    let dt = 0.5 * dx / c;

    let nu = c * dt / (2.0 * dx);

    let mut u_current = vec![0.0; num_space_points + 2];
    let mut u_next = u_current.clone();

    let x_values: Vec<f64> = (0..=num_space_points + 1)
        .map(|i| (i as f64 - 1.0) * dx)
        .collect();

    let mut snapshots = Vec::new();
    let snapshot_times = vec![0.5, 1.0, 1.5, 2.0];

    let mut time = 0.0;


    let mut snapshot_index = 0;

    while time < t_max {
        time += dt;

        u_current[1] = (-time).exp();

        for i in 1..=num_space_points {
            u_next[i] = u_current[i] - nu * (u_current[i + 1] - u_current[i - 1]);
        }

        u_next[0] = u_next[1];
        u_next[num_space_points + 1] = u_next[num_space_points];

        u_current.copy_from_slice(&u_next);

        let epsilon = 1e-6;
        if snapshot_index < snapshot_times.len()
            && (time - snapshot_times[snapshot_index]).abs() < epsilon
        {
            snapshots.push((time, u_current.clone()));
            snapshot_index += 1;
        }

        // snapshots.push((time, u_current.clone()));
    }

    for (time, u) in snapshots {
        let u_numerical = &u[1..=num_space_points + 1];
        let x_plot = &x_values[1..=num_space_points + 1];
        println!("Time = {}", time);
        plot(x_plot, u_numerical, u_numerical);
    }
}
