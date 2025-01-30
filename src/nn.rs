use crate::alloc::Alloc;
use serde::{Deserialize, Serialize};
use std::iter::once;
use std::mem::swap;

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
    pub fn run(&self, inp: &mut Vec<f64>, res: &mut Vec<Vec<f64>>, alloc: &mut Alloc) {
        let mut next = alloc.alloc();
        for layer in self.layers.iter() {
            next.clear();
            for n in layer {
                let mut val = 0.;
                assert_eq!(inp.len() + 1, n.len());
                for (i, w) in inp.iter().chain(once(&1.)).zip(n.iter()) {
                    val += i * w;
                }
                next.push(val);
            }
            let mut buf = alloc.alloc();
            buf.clone_from(&next);
            res.push(buf);
            for i in next.iter_mut() {
                *i = f(*i);
            }
            swap(inp, &mut next);
        }
        alloc.dealloc(next)
    }
    pub fn train(&mut self, dt: &mut Vec<f64>, log: Vec<Vec<f64>>, alloc: &mut Alloc) {
        let mut next = alloc.alloc();
        for (layer, inps) in self.layers.iter_mut().zip(log).rev() {
            next.clear();
            let l = layer[0].len();
            next.reserve_exact(l);
            for _ in 0..l {
                next.push(0.);
            }
            assert_eq!(layer.len(), inps.len());
            assert_eq!(inps.len(), dt.len());
            for ((n, i), d) in layer.iter_mut().zip(&inps).zip(dt.iter()) {
                let d = dfr(*i) * d;
                let wsum = n.iter().map(|v| v.abs()).sum::<f64>();
                for (idx, w) in n.iter_mut().enumerate() {
                    let cont = *w / wsum;
                    *w += d * cont;
                    next[idx] += d * cont;
                }
            }
            swap(dt, &mut next);
            dt.pop(); // bias
            alloc.dealloc(inps)
        }
        alloc.dealloc(next)
    }
}
