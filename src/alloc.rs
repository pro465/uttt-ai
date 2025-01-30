pub struct Alloc(Vec<Vec<f64>>);

impl Alloc {
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn alloc(&mut self) -> Vec<f64> {
        self.0.pop().unwrap_or_default()
    }
    pub fn dealloc(&mut self, mut v: Vec<f64>) {
        v.clear();
        self.0.push(v)
    }

    pub fn dealloc_bulk(&mut self, v: &mut Vec<Vec<f64>>) {
        for v in v.iter_mut() {
            v.clear();
        }
        self.0.append(v)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}
