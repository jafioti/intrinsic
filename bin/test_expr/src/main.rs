use intrinsic::*;

fn main() {
    let now = std::time::Instant::now();
    let orig = (expr('z')
        / ((-5 + (((((-5 + ((((((expr('w') + 153) / 2) / 2) / 2) / 2) / 2)) * 4) + 9) / 2) / 2))
            * (-5
                + (((9 + (4 * (-5 + ((((((153 + expr('h')) / 2) / 2) / 2) / 2) / 2)))) / 2) / 2))))
        % 64;
    let intermediate = egglog_simplify(orig, 5, 5);
    println!("{intermediate} | {}", intermediate.len());
    println!("Took {}s", now.elapsed().as_secs_f32());
    let now = std::time::Instant::now();
    let r = egglog_simplify(orig, 8, 1);
    println!("{r} | {}", r.len());
    println!("Took {}s", now.elapsed().as_secs_f32());
    expression_cleanup();
}
