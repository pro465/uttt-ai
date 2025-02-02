use crate::alloc::Alloc;
pub type PlayerId = usize;

pub type Cell = Option<PlayerId>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Ongoing,
    Draw,
    Won(PlayerId),
}

impl GameResult {
    pub fn is_ongoing(&self) -> bool {
        matches!(self, GameResult::Ongoing)
    }
    pub fn is_draw(&self) -> bool {
        matches!(self, GameResult::Draw)
    }
    pub fn is_won(&self) -> bool {
        matches!(self, GameResult::Won(_))
    }
}

pub type Move = (usize, usize);

pub const LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

#[derive(Clone, Debug)]
pub struct SubSquare {
    board: Vec<Cell>,
    state: GameResult,
}

impl SubSquare {
    pub fn new() -> Self {
        Self {
            board: vec![None; 9],
            state: GameResult::Ongoing,
        }
    }

    pub fn get2(&self, i: usize, j: usize) -> Cell {
        self.get1(3 * i + j)
    }
    pub fn get1(&self, i: usize) -> Cell {
        self.board[i]
    }
    pub fn put(&mut self, i: usize, v: PlayerId) {
        assert!(self.state.is_ongoing());
        self.board[i] = Some(v);
        self.analyze();
    }
    pub fn reset(&mut self, i: usize) {
        self.board[i] = None;
        self.state = GameResult::Ongoing;
    }
    pub fn state(&self) -> GameResult {
        self.state
    }
    fn analyze(&mut self) {
        for l in LINES {
            let [a, b, c] = l.map(|i| self.get1(i));
            if a == b && b == c {
                if let Some(a) = a {
                    self.state = GameResult::Won(a);
                    return;
                }
            }
        }
        self.state = if self.board.iter().all(|i| i.is_some()) {
            GameResult::Draw
        } else {
            GameResult::Ongoing
        }
    }
}

#[derive(Clone, Debug)]
pub struct Square {
    board: Vec<SubSquare>,
    prev: Option<usize>,
}

impl Square {
    pub fn new() -> Self {
        Self {
            board: vec![SubSquare::new(); 9],
            prev: None,
        }
    }

    pub fn prev(&self) -> Option<usize> {
        self.prev
    }
    pub fn get2(&self, i: usize, j: usize) -> &SubSquare {
        self.get1(3 * i + j)
    }
    pub fn get1(&self, i: usize) -> &SubSquare {
        &self.board[i]
    }
    pub fn put(&mut self, i: usize, si: usize, v: PlayerId) {
        self.board[i].put(si, v);
        self.prev = Some(si)
    }
    pub fn is_valid1(&self, x: usize) -> bool {
        self.get1(x).state().is_ongoing()
            && match self.prev {
                None => true,
                Some(y) => x == y,
            }
    }
    pub fn is_valid2(&self, x: usize, y: usize) -> bool {
        self.get1(x).state().is_ongoing()
            && self.get1(x).get1(y).is_none()
            && match self.prev {
                None => true,
                Some(z) => x == z,
            }
    }
    pub fn valid_moves(&self, alloc: &mut Alloc) -> Vec<Move> {
        let mut moves = alloc.alloc_mvec();
        for i in 0..9 {
            if !self.is_valid1(i) {
                continue;
            }
            for j in 0..9 {
                if self.is_valid2(i, j) {
                    moves.push((i, j));
                }
            }
        }
        moves
    }

    pub fn reset(&mut self, prev: Option<usize>, x: usize, y: usize) {
        self.prev = prev;
        self.board[x].reset(y);
    }

    pub fn analyze(&mut self) -> GameResult {
        if let Some(x) = self.prev {
            self.board[x].analyze();
            if !self.board[x].state().is_ongoing() {
                self.prev = None;
            }
        }
        for l in LINES {
            let [a, b, c] = l.map(|i| self.get1(i).state());
            if a.is_won() && a == b && b == c {
                return a;
            }
        }
        if self.board.iter().any(|i| i.state().is_ongoing()) {
            //dbg!(&self.board);
            GameResult::Ongoing
        } else {
            GameResult::Draw
        }
    }

    pub fn feedback(&self, p: PlayerId) -> f64 {
        let mut res = 0.;
        for s in self.board.iter() {
            if let GameResult::Won(i) = s.state() {
                res += if i == p { 0.05 } else { -0.05 };
            }
        }
        res
    }
}
use std::fmt;

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..9 {
            if i % 3 == 0 && i > 0 {
                write!(f, "----+-----+----\n")?;
            }
            for j in 0..9 {
                if j % 3 == 0 && j > 0 {
                    write!(f, " | ")?;
                }
                let sq = self.get2(i / 3, j / 3);
                let c = sq.get2(i % 3, j % 3);
                match c {
                    None => write!(f, " ")?,
                    Some(i) => write!(f, "{}", ['X', 'O'][i])?,
                }
            }
            write!(f, "\n")?
        }
        Ok(())
    }
}
