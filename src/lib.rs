pub mod alloc;
pub mod game;
pub mod nn;
pub mod rl;
pub mod storage;

use rand::prelude::*;

pub fn random_nn(range: (f64, f64), layout: &[usize]) -> nn::NN {
    let mut v = Vec::new();
    let mut rng = rand::rng();
    let (a, b) = range;
    for (&m, &n) in layout.iter().zip(layout.iter().skip(1)) {
        let m = m + 1;
        let mf = m as f64;
        let (an, bn) = (a / mf, b / mf);
        v.push(
            (0..n)
                .map(|_| (0..m).map(|_| rng.random_range(an..=bn)).collect())
                .collect(),
        );
    }
    nn::NN::new(v)
}
