fn factolu(l: Vec<f64>, d: Vec<f64>, u: Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = d.len();
    assert_eq!(n - 1, l.len());
    assert_eq!(n - 1, u.len());
    let up = u;
    let mut lp = vec![0.; n - 1];
    let mut dp = vec![0.; n];
    dp[0] = d[0];

    for i in 0..n - 1 {
        lp[i] = l[i] / dp[i];
        dp[i + 1] = d[i+1] - lp[i] * up[i];
    }
    (lp, dp, up)
}

fn main() {
    let l = vec![3., 14., 3.];
    let u = vec![1., 1., 1.];
    let d = vec![1., 5., 10., 5.];

    let (lp,dp,up) = factolu(l,d,u);

    println!("l={:?}", lp);
    println!("d={:?}", dp);
    println!("u={:?}", up);

}
