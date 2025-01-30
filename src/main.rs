use rl::RLer;
use std::io::*;
use uttt::*;

use alloc::Alloc;

const REC_LVL: u64 = 2;

fn main() {
    let mut rler = get_model();
    let mut alloc = Alloc::new();
    train_model(&mut rler, &mut alloc);
    let (p, min_p)=(rler.p, rler.min_p);
    rler.p=0.;
    rler.min_p=0.;
    while play_with_model(&mut rler, &mut alloc) {}
    rler.p=p;
    rler.min_p=min_p;
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
    let sp = random_nn((-1.0, 1.0), &[36, 6, 1]);
    RLer {
        first_pass: fp,
        second_pass: sp,
        p: 0.9,
        p_decay: 0.995,
        min_p: 0.01,
        lr: 0.05,
        lr_decay: 0.999,
        imp_decay: 0.9,
        min_lr: 0.001,
    }
}

fn train_model(rler: &mut RLer, alloc: &mut Alloc) {
    let s = input("train the model? (y/n): ");
    if !["Y", "y"].contains(&s.trim()) {
        return;
    }
    let n = input("number of training steps: ")
        .trim()
        .parse()
        .expect("not an integer");
    for i in 0..n {
        let (g, gr) = rler.gen_game_for_training(REC_LVL, alloc);
        println!("{i} {:?} {}", rler.train(g.clone(), gr, alloc), g.len());
    }
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
