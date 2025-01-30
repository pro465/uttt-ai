use crate::game::{GameResult, Move, PlayerId, Square};
use crate::nn::NN;
use crate::alloc::Alloc;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

pub type RLResult = (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, f64);

#[derive(Debug, Deserialize, Serialize)]
pub struct RLer {
    pub first_pass: NN,
    pub second_pass: NN,
    pub p: f64,
    pub p_decay: f64,
    pub min_p: f64,
    pub lr: f64,
    pub lr_decay: f64,
    pub imp_decay: f64,
    pub min_lr: f64,
}

impl RLer {
    pub fn run(&self, game: &Square, perspective: PlayerId, alloc: &mut Alloc) -> RLResult {
        let mut log = Vec::new();
        let mut l = Vec::new();
        let mut input1 = alloc.alloc();
        let mut input2 = alloc.alloc();
        for i in 0..9 {
            for j in 0..9 {
                let c = game.get1(i).get1(j);
                let c = match c {
                    None => 0.,
                    Some(p) if p == perspective => 1.,
                    _ => -1.,
                };
                input1.push(c);
            }
            self.first_pass.run(&mut input1, &mut l, alloc);
            input2.append(&mut input1);
            log.push(l);
            l=Vec::new();
        }
        alloc.dealloc(input1);
        for i in 0..9 {
            input2.push([0., 1.][game.is_valid1(i) as usize]);
        }
        self.second_pass.run(&mut input2, &mut l, alloc);
        let res = input2[0];
        alloc.dealloc(input2);
        (log, l, res)
    }

    pub fn train(&mut self, game: Vec<Square>, game_res: GameResult, alloc: &mut Alloc) -> GameResult {
        let cost = match game_res {
                GameResult::Won(i) => [1., -1.][i],
                GameResult::Draw => return game_res,
                _ => unreachable!(),
            };
        let mut lr = self.lr;
        let mut d =alloc.alloc();
        let mut d2=alloc.alloc();
        for g in game.iter().rev() {
            for p in 0..2 {
                let fb = g.feedback(p);
                let (log, l, o) = self.run(g, p, alloc);
                let c = fb + [cost, -cost][p] - o;
                d.push(c*lr);
                self.second_pass.train(&mut d, l, alloc);
                for (i, j) in d.chunks_exact(d.len() / 9).zip(log) {
                    d2.extend_from_slice(i);
                    self.first_pass.train(&mut d2, j, alloc);
                    d2.clear();
                }
                d.clear();
            }
            lr *= self.imp_decay;
        }
        alloc.dealloc(d);
        alloc.dealloc(d2);
        self.lr *= self.lr_decay;
        self.p *= self.p_decay;
        self.lr = self.lr.max(self.min_lr);
        self.p = self.p.max(self.min_p);
        game_res
    }

    pub fn gen_game_for_training(&self, recursion_depth: u64, alloc: &mut Alloc) -> (Vec<Square>, GameResult) {
        let mut g = Square::new();
        let mut v = Vec::new();
        let mut p = 0;
        while g.analyze().is_ongoing() {
            v.push(g.clone());
            let (x, y) = self.gen_move(&g, p, recursion_depth, alloc).1;
            g.put(x, y, p);
            p ^= 1;
        }
        v.push(g.clone());
        (v, g.analyze())
    }

    pub fn gen_move(&self, game: &Square, player: PlayerId, rem_depth: u64, alloc: &mut Alloc) -> (f64, Move) {
        let moves = game.valid_moves();
        let mut g = game.clone();
            let mut sc = |g: &mut Square| if rem_depth == 0 || !g.analyze().is_ongoing() {
                let (log, l, o)=self.run(&g, player, alloc);
                for mut i in log { alloc.dealloc_bulk(&mut i); }
                for i in l { alloc.dealloc(i); }
                o
            } else  {
                -self.gen_move(&g, 1-player, rem_depth-1, alloc).0
            };
        let mut rng = rand::rng();

        if rng.random_bool(self.p) {
            let (x, y) = *moves.choose(&mut rng).unwrap();
            let p = game.prev();
            g.put(x, y, player);
            let sc = sc(&mut g);
            g.reset(p, x, y);
            return (sc, (x, y));
        }
        let (mut bsc, mut bm) = (f64::NEG_INFINITY, (9, 9));
        for (x, y) in moves {
            let p = game.prev();
            g.put(x, y, player);
            let sc = sc(&mut g);
            if sc > bsc {
                bsc = sc;
                bm = (x, y);
            }
            g.reset(p, x, y);
        }
        (bsc, bm)
    }
}
