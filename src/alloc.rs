use crate::game::Move;

pub struct Alloc(Vec<Vec<f64>>, Vec<Vec<Move>>);

impl Alloc {
    pub fn new() -> Self {
        Self(Vec::new(), Vec::new())
    }
    pub(crate) fn alloc(&mut self) -> Vec<f64> {
        self.0.pop().unwrap_or_default()
    }
    pub(crate) fn dealloc(&mut self, mut v: Vec<f64>) {
        v.clear();
        self.0.push(v)
    }

    pub(crate) fn dealloc_bulk(&mut self, v: &mut Vec<Vec<f64>>) {
        for v in v.iter_mut() {
            v.clear();
        }
        self.0.append(v)
    }

    pub(crate) fn alloc_mvec(&mut self) -> Vec<Move> {
        self.1.pop().unwrap_or_default()
    }
    pub(crate) fn dealloc_mvec(&mut self, mut v: Vec<Move>) {
        v.clear();
        self.1.push(v)
    }


    pub fn len(&self) -> usize {
        self.0.len()
    }
}
