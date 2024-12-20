// masse de Dirac approchée
fn delta(t: f64, tau: f64) -> f64 {
    let s = t / tau;
    let d = if -0.5 < s && s < 0.5 { 1.0 } else { 0.0 };
    d / tau
}

// condition initiale de classe C^2
// à support compact
fn u0(x: f64) -> f64 {
    let sigma = 140.;
    let x0 = 1.;
    let y = x - x0;
    let v = if y.abs() < 0.5 {
        sigma * (0.5 - y).powf(3.) * (y + 0.5).powf(3.)
    } else {
        0.
    };
    v
}

// solution exacte
fn uexact(x: f64, t: f64) -> f64 {
    let c = 1.;
    if t < 0. || x < 0. {
        0.
    } else {
        if x < c * t {
            g(t - x / c)
        } else {
            u0(x - c * t)
        }
    }
}

// condition en temps à gauche
fn g(t: f64) -> f64 {
    let v = (-t).exp();
    if t <= 0. {
        0.
    } else {
        v
    }
}

fn solve_exp_order1(tmin: f64, tmax:f64, xmin: f64, xmax: f64, cfl: f64, c: f64, nmax: usize) -> f64 {

    let h = (xmax - xmin) / (nmax as f64);
    let dt = cfl * h / c;

    let xi = (0..nmax)
        .map(|i| xmin + (i as f64) * h)
        .collect::<Vec<f64>>();

    // vérifie qu'un seul xi[k] vérifie delta(xi[k],h) != 0
    let mut count = 0;
    for i in 0..nmax {
        if delta(xi[i], h) != 0. {
            count += 1;
            println!("xi[{}] = {}", i, xi[i]);
        }
    }
    assert_eq!(count, 1);

    let mut un = vec![0.; nmax];
    let mut unp1 = un.clone();

    let mut t = tmin;

    while t < tmax {
        t += dt;
        //println!("t = {}", t);
        for i in 1..nmax {
            unp1[i] = un[i] - c * dt / h * (un[i] - un[i - 1])
                + dt * delta(t, dt) * u0(xi[i])
                + dt * c * delta(xi[i], h) * g(t);
        }
        for i in 0..nmax {
            un[i] = unp1[i];
        }
    }
    println!("tmax = {}, t={}", tmax, t);

    let uex = (0..nmax).map(|i| uexact(xi[i], t)).collect::<Vec<f64>>();

    // compute L2 norm, but on [0,xmax] only
    let mut errl2 = 0.;
    for i in 0..nmax {
        if xi[i] > 0. {
        errl2 += h*(un[i] - uex[i]).powi(2);
        }
    }

    errl2 = errl2.sqrt();

    use rsplot1d::plot;
    plot(&xi, &un, &uex);
    errl2
}

fn main(){
    let tmax = 1.;
    let tmin = -0.1;

    let xmin = -1.;
    let xmax = 3.;

    let cfl = 0.7;

    let c = 1.0;

    let nmax = 3000;

    let errl2 = solve_exp_order1(tmin, tmax, xmin, xmax, cfl, c, nmax);

    println!("L2 error = {}", errl2);

}

// dt w + dx (q(w)) = delta * (R(wbar,w0,x/t=0)-wbar)

// f <- 2 feq - f 