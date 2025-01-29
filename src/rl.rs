use crate::game::{GameResult, Move, PlayerId, Square};
use crate::nn::NN;
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
    pub fn run(&self, game: &Square, perspective: PlayerId) -> RLResult {
        let mut log = Vec::new();
        let mut input2 = Vec::new();
        for i in 0..9 {
            let mut input1 = Vec::new();
            for j in 0..9 {
                let c = game.get1(i).get1(j);
                let c = match c {
                    None => 0.,
                    Some(p) if p == perspective => 1.,
                    _ => -1.,
                };
                input1.push(c);
            }
            let (l, mut o) = self.first_pass.run(input1.clone());
            input2.append(&mut o);
            log.push(l);
        }
        let (l, o) = self.second_pass.run(input2);
        (log, l, o[0])
    }

    pub fn train(&mut self, game: Vec<Square>, game_res: GameResult) -> GameResult {
        let cost = match game_res {
                GameResult::Won(i) => [1., -1.][i],
                GameResult::Draw => return game_res,
                _ => unreachable!(),
            };
        let mut lr = self.lr;
        for g in game.iter().rev() {
            for p in 0..2 {
                let fb = g.feedback(p);
                let (log, l, o) = self.run(g, p);
                let c = fb + [cost, -cost][p] - o;
                let d = self.second_pass.train(vec![c * lr], l);
                for (i, j) in d.chunks_exact(d.len() / 9).zip(log.into_iter()) {
                    self.first_pass.train(i.to_vec(), j);
                }
            }
            lr *= self.imp_decay;
        }
        self.lr *= self.lr_decay;
        self.p *= self.p_decay;
        self.lr = self.lr.max(self.min_lr);
        self.p = self.p.max(self.min_p);
        game_res
    }

    pub fn gen_game_for_training(&self) -> (Vec<Square>, GameResult) {
        let mut g = Square::new();
        let mut v = Vec::new();
        let mut p = 0;
        while g.analyze().is_ongoing() {
            v.push(g.clone());
            let (x, y) = self.gen_move(&g, p, false);
            g.put(x, y, p);
            p ^= 1;
        }
        v.push(g.clone());
        (v, g.analyze())
    }

    pub fn gen_move(&self, game: &Square, player: PlayerId, db: bool) -> Move {
        let moves = game.valid_moves();
        let mut rng = rand::rng();

        if db {
            println!("{}", &game);
        }
        if rng.random_bool(self.p) {
            if db {
                println!("random");
            }
            return *moves.choose(&mut rng).unwrap();
        }
        let (mut bsc, mut bm) = (f64::NEG_INFINITY, (9, 9));
        let mut g = game.clone();
        for (x, y) in moves {
            let p = game.prev();
            g.put(x, y, player);
            let sc = self.run(&g, player).2;
            if db {
                dbg!(x, y, sc);
            }
            if sc > bsc {
                bsc = sc;
                bm = (x, y);
            }
            g.reset(p, x, y);
        }
        bm
    }
}
