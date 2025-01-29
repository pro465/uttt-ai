use serde::{Deserialize, Serialize};
use std::iter::once;

fn f(x: f64) -> f64 {
    assert!(!x.is_nan());
    x.tanh()
}

// dx/df
fn dfr(x: f64) -> f64 {
    assert!(!x.is_nan());
    let t = x.cosh();
    t * t
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NN {
    layers: Vec<Vec<Vec<f64>>>,
}

impl NN {
    pub fn new(layers: Vec<Vec<Vec<f64>>>) -> Self {
        Self { layers }
    }
    pub fn run(&self, mut inp: Vec<f64>) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut res = vec![inp.clone()];
        let mut next = Vec::new();
        for layer in self.layers.iter() {
            next.clear();
            for n in layer {
                let mut val = 0.;
                for (i, w) in inp.iter().chain(once(&1.)).zip(n.iter()) {
                    val += i * w;
                }
                next.push(val);
            }
            res.push(next.clone());
            for i in next.iter_mut() {
                *i = f(*i);
            }
            (next, inp) = (inp, next);
        }
        (res, inp)
    }
    pub fn train(&mut self, mut dt: Vec<f64>, log: Vec<Vec<f64>>) -> Vec<f64> {
        let mut next = Vec::new();
        for (layer, inps) in self.layers.iter_mut().zip(&log).rev() {
            next.clear();
            let l = layer[0].len();
            next.reserve_exact(l);
            for _ in 0..l {
                next.push(0.);
            }
            for ((n, i), d) in layer.iter_mut().zip(inps).zip(&dt) {
                let d = dfr(*i) * d;
                let wsum = n.iter().map(|v| v.abs()).sum::<f64>();
                for (idx, w) in n.iter_mut().enumerate() {
                    let cont = *w / wsum;
                    *w += d * cont;
                    next[idx] += d * cont;
                }
            }
            (dt, next) = (next, dt);
        }
        dt
    }
}
