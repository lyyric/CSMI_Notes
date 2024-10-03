// résolution de l'équation de la chaleur en 1D
// par 1) méthode de développement en série
// 2) méthode des différences finies
fn main() {
    let nx = 1000;

    let h = 1.0 / nx as f64;
    let cfl = 0.9;
    let dt = 0.5 * h * h * cfl;

    let tmax = 0.001;

    let mut us = vec![0.25; nx];

    let xi = (0..nx).map(|i| i as f64 * h + h / 2.).collect::<Vec<f64>>();

    let pi = std::f64::consts::PI;

    let sin = f64::sin;
    let exp = f64::exp;
    let cos = f64::cos;

    let nk = 10;
    for k in 1..nk + 1 {
        let a = 0.5 - 1. / 8 as f64;
        let b = 0.5 + 1. / 8 as f64;
        let ck = 2. / k as f64 / pi * (sin(k as f64 * pi * b) - sin(k as f64 * pi * a));
        let ee = exp(-k as f64 * k as f64 * pi * pi * tmax);
        for i in 0..nx {
            us[i] += ck * ee * cos(k as f64 * pi * xi[i]);
        }
    }

    use rsplot1d::plot;
    plot(&xi, &us, &us);

    use skyrs::Sky;
    let mut coo = vec![];

    coo.push((0, 0, 1. - cfl / 2.));
    coo.push((nx - 1, nx - 1, 1. - cfl / 2.));
    for i in 1..nx - 1 {
        coo.push((i, i, 1. - cfl));
    }
    for i in 0..nx - 1 {
        coo.push((i, i + 1, cfl / 2.));
        coo.push((i + 1, i, cfl / 2.));
    }

    let sky = Sky::new(coo);

    //let mut un = xi.iter().map(|&x| uinit(x)).collect::<Vec<f64>>();
    let mut un = vec![0.; nx];
    for i in 0..nx {
        un[i] = uinit(xi[i]);
    }
    let mut t = 0.;
    while t < tmax {
        un = sky.dot(&un);
        t += dt;
    }
    plot(&xi, &us, &un);
}

fn uinit(x: f64) -> f64 {
    if x < 0.5 - 1. / 8. || x > 0.5 + 1. / 8. {
        0.
    } else {
        1.
    }
}