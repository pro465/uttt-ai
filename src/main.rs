use game::GameResult;
use rl::RLer;
use std::io::*;
use uttt::*;
use rand::prelude::*;
use alloc::Alloc;

const REC_LVL: u64 = 2;

fn main() {
    let mut rler = get_model();
    let mut alloc = Alloc::new();
    train_model(&mut rler, &mut alloc);
    let (p, min_p) = (rler.p, rler.min_p);
    rler.p = 0.;
    rler.min_p = 0.;
    while play_with_model(&mut rler, &mut alloc) {}
    rler.p = p;
    rler.min_p = min_p;
    save_model(rler);
}

fn input(prompt: &str) -> String {
    let mut stdout = stdout();
    write!(stdout, "{}", prompt).unwrap();
    stdout.flush().unwrap();
    let mut buffer = String::new();
    stdin().read_line(&mut buffer).unwrap();
    buffer
}

fn get_model() -> RLer {
    let s = input("load existing model? (y/n): ");
    if ["Y", "y"].contains(&s.trim()) {
        let fname = input("file name: ");
        storage::load(fname.trim()).expect("an error occured")
    } else {
        random_model()
    }
}

fn random_model() -> RLer {
    let fp = random_nn((-1.0, 1.0), &[9, 5, 3]);
    let sp = random_nn((-1.0, 1.0), &[37, 15, 5, 1]);
    RLer {
        first_pass: fp,
        second_pass: sp,
        p: 1.,
        p_decay: 0.995,
        min_p: 0.05,
        lr: 0.001,
        lr_decay: 0.99,
        imp_decay: 1.,
        min_lr: 0.0001,
    }
}

fn train_model(m1: &mut RLer, alloc: &mut Alloc) {
    let s = input("train the model? (y/n): ");
    if !["Y", "y"].contains(&s.trim()) {
        return;
    }
    println!("for the adversarial model:");
    let mut models = Vec::new();
    loop {
        models.push(get_model());
        let s = input("add another adversarial model? (y/n): ");
        if !["Y", "y"].contains(&s.trim()) { break }
    }
    let n_games = input("number of games: ")
        .trim()
        .parse()
        .expect("not an integer");

    let n_steps = input("number of training steps for each game: ")
        .trim()
        .parse()
        .expect("not an integer");

    let mut rng = rand::rng();
    let (mut wins, mut losses, mut draws) = (0, 0, 0);
    for i in 0..n_games {
        let m2 = models.choose_mut(&mut rng).expect("number of models is always non-zero");
        let (first, sec) = [(&*m1, &*m2), (m2, m1)][i&1];
        let (g, gr) = first.gen_game_for_training(sec, REC_LVL, alloc);
        println!("game {i}: result: {gr:?}\n=========================");
        match gr {
            GameResult::Won(p) => *[&mut wins, &mut losses][(p ^ i) & 1] += 1,
            GameResult::Draw => draws += 1,
            GameResult::Ongoing => unreachable!(),
        }

        for j in 0..n_steps {
            m1.train(g.clone(), gr, alloc);
            m2.train(g.clone(), gr, alloc);
        }

    }
    println!("wins: {wins}\nlosses: {losses}\ndraws: {draws}");
}

fn save_model(model: RLer) {
    let s = input("save the model? (y/n): ");
    if !["Y", "y"].contains(&s.trim()) {
        return;
    }

    let fname = input("file name: ");
    storage::save(model, fname.trim()).expect("an error occured");
}


fn play_with_model(model: &mut RLer, alloc: &mut Alloc) -> bool {
    let s = input("play with the model? (will train the model too) (y/n): ");
    if !["Y", "y"].contains(&s.trim()) {
        return false;
    }
    let human = input("wanna go first? (y/n): ");
    let human = if ["Y", "y"].contains(&human.trim()) {
        0
    } else {
        1
    };
    let mut current_turn = 0;
    let mut g = game::Square::new();
    let mut v = Vec::new();
    while g.analyze().is_ongoing() {
        println!("{}", &g);
        v.push(g.clone());
        let is_human = current_turn == human;
        let (x, y) = if is_human {
            get_move(&g)
        } else {
            model.gen_move(&g, current_turn, REC_LVL, alloc).1
        };
        println!(
            "{} chose subsquare {}, cell {}",
            ["AI", "You"][is_human as usize],
            x + 1,
            y + 1
        );

        g.put(x, y, current_turn);
        current_turn ^= 1;
    }
    println!("{}", &g);
    v.push(g.clone());
    model.train(v, g.analyze(), alloc);
    true
}

fn get_move(g: &game::Square) -> game::Move {
    let x = loop {
        let x = input("choose subsquare: ").trim().parse().unwrap_or(0);
        if 0 < x && x < 10 && g.is_valid1(x - 1) {
            break x - 1;
        }
        println!("invalid subsquare address.");
    };
    let y = loop {
        let y = input("choose cell: ").trim().parse().unwrap_or(0);
        if 0 < y && y < 10 && g.is_valid2(x, y - 1) {
            break y - 1;
        }
        println!("invalid cell address.");
    };
    (x, y)
}
