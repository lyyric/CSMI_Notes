#[allow(dead_code)]
pub fn run() {
    use rsplot1d::plot;

    let space_points_list = vec![50, 100, 200, 400, 800];

    let mut l1_errors = Vec::new();
    let mut l2_errors = Vec::new();
    let mut dx_values = Vec::new();

    for &num_space_points in &space_points_list {
        let dx = 1.0 / num_space_points as f64;
        let c = 1.0;

        let t_max = 0.5;
        let cfl_number = 0.8;
        let dt = cfl_number * dx / c;
        let v = c * dt / dx;

        let mut u_current = vec![0.0; num_space_points + 1];
        let mut u_next = u_current.clone();

        let x_values: Vec<f64> = (0..=num_space_points)
            .map(|i| i as f64 * dx)
            .collect();

        let mut time = 0.0;

        while time < t_max {

            time += dt;

            u_next[0] = (-time).exp();

            for i in 1..=num_space_points {
                if i < num_space_points {
                    u_next[i] = (1.0 - v) * u_current[i] + v * u_current[i - 1];
                } else {
                    u_next[i] = 0.0;
                }
            }

            u_current.copy_from_slice(&u_next);
        }

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

        let mut l1_error = 0.0;
        let mut l2_error = 0.0;

        for i in 0..=num_space_points {
            let error = (u_current[i] - exact_solution[i]).abs();
            l1_error += error * dx;
            l2_error += error * error * dx;
        }

        l2_error = l2_error.sqrt();

        l1_errors.push(l1_error);
        l2_errors.push(l2_error);
        dx_values.push(dx);
    }

    println!("dx\tL1 Error\tL2 Error");
    for i in 0..dx_values.len() {
        println!(
            "{:.5}\t{:.5e}\t{:.5e}",
            dx_values[i], l1_errors[i], l2_errors[i]
        );
    }

    let dx_values: Vec<f64> = dx_values.iter().map(|&dx| dx.ln()).collect();
    let l1_errors: Vec<f64> = l1_errors.iter().map(|&err| err.ln()).collect();
    let l2_errors: Vec<f64> = l2_errors.iter().map(|&err| err.ln()).collect();


    plot(&dx_values, &l1_errors, &l2_errors);

}
