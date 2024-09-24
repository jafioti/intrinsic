mod ir;

use intrinsic::{egglog_simplify, expression_cleanup, Expression};
use ir::*;

fn main() {
    let b_graph = block(3, (Tensor("A1".to_string()), 1), exp(refr(0)));
    let (kernel, grid, threadblock) = codegen(&b_graph);
    println!("B");
    println!("------------");
    println!("{kernel}");
    println!("{grid:?}");
    println!("{threadblock:?}");
    let graph = block(
        4,
        (Tensor("A".to_string()), 15),
        sum(block(
            5,
            (Ref(0), 3),
            block(
                3,
                ((Ref(0), 1), (Tensor("B".to_string()), 1)),
                exp(sin(mul(refr(0), refr(1)))),
            ),
        )),
    );
    let (kernel, grid, threadblock) = codegen(&graph);
    println!("C");
    println!("------------");
    println!("{kernel}");
    println!("{grid:?}");
    println!("{threadblock:?}");
    expression_cleanup();
}

fn codegen(block: &Block) -> (String, (u32, u32, u32), (u32, u32, u32)) {
    let (kernel, var, mut grid) = inner_codegen(block, 0, 0, false, &[]);
    grid.append(&mut vec![1; 6 - grid.len()]);
    let input_string = get_input_tensors(block)
        .into_iter()
        .map(|t| format!("const float *{t}"))
        .collect::<Vec<_>>()
        .join(", ");
    let kernel = format!(
        "extern \"C\" __global__ void kernel(float *out{}) {{
{kernel}
\tout[{}] = {var};
}}",
        if !input_string.is_empty() {
            format!(", {input_string}")
        } else {
            "".to_string()
        },
        grid_setter_idx(&grid)
    );
    (
        kernel,
        (grid[0], grid[1], grid[2]),
        (grid[3], grid[4], grid[5]),
    )
}

fn inner_codegen(
    block: &Block,
    grid_level: u8,
    iter_level: u8,
    is_iter: bool,
    tensors: &[(String, Expression)],
) -> (String, String, Vec<u32>) {
    // Add new stride term to the index expression
    let tensors = block
        .1
        .iter()
        .map(|(inp, stride)| match inp {
            Input::Ref(Ref(r)) => (
                tensors[*r as usize].0.clone(),
                tensors[*r as usize].1
                    + Expression::from(if is_iter {
                        format!("iter_{}", (iter_level + 96) as char)
                    } else {
                        grid_levels(grid_level).to_string()
                    }) * *stride as i32,
            ),
            Input::Tensor(Tensor(t)) => (
                t.clone(),
                Expression::from(if is_iter {
                    format!("iter_{}", (iter_level + 96) as char)
                } else {
                    grid_levels(grid_level).to_string()
                }) * *stride as i32,
            ),
        })
        .collect::<Vec<_>>();
    match block.2.as_ref() {
        Body::Block(inner) => {
            let (body, var, mut grid) =
                inner_codegen(inner, grid_level + 1, iter_level, false, &tensors);
            if !is_iter {
                grid.insert(0, block.0 as u32);
            }
            (body, var, grid)
        }
        Body::Op(op) => {
            let op = gen_op(op, &tensors);
            (
                format!("\tfloat val = {op}"),
                "val".to_string(),
                if is_iter {
                    vec![]
                } else {
                    vec![block.0 as u32]
                },
            )
        }
        Body::Sum(inner) => {
            let iter_char = (iter_level + 97) as char;
            let (kernel, var, mut grid) =
                inner_codegen(inner, grid_level, iter_level + 1, true, &tensors);
            let tabs = (0..iter_level + 1)
                .map(|_| "\t")
                .collect::<Vec<_>>()
                .join("");
            let kernel = format!(
                "{tabs}float acc_{iter_char} = 0.0;
{tabs}for (int iter_{iter_char} = 0; iter_{iter_char} < {}; iter_{iter_char}++) {{
{tabs}{kernel}
{tabs}\tacc_{iter_char} += {var};
{tabs}}}",
                get_sum_size(inner)
            );
            if !is_iter {
                grid.insert(0, block.0 as u32);
            }
            (kernel, format!("acc_{iter_char}"), grid)
        }
    }
}

fn gen_op(op: &Op, tensors: &[(String, Expression)]) -> String {
    match op {
        Op::Add(a, b) => format!("{} + {}", gen_op(a, tensors), gen_op(b, tensors)),
        Op::Mul(a, b) => format!("{} * {}", gen_op(a, tensors), gen_op(b, tensors)),
        Op::Exp(a) => format!("exp({})", gen_op(a, tensors)),
        Op::Sin(a) => format!("sin({})", gen_op(a, tensors)),
        Op::Ref(a) => {
            format!(
                "{}[{}]",
                tensors[*a as usize].0,
                egglog_simplify(tensors[*a as usize].1, 3, 1)
            )
        }
    }
}

fn grid_levels(level: u8) -> &'static str {
    [
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "threadIdx.x",
        "threadIdx.y",
        "threadIdx.z",
    ][level as usize]
}

// Get the input tensors of a block structure
fn get_input_tensors(block: &Block) -> Vec<String> {
    let inps = block
        .1
        .iter()
        .filter_map(|(i, _)| {
            if let Input::Tensor(Tensor(t)) = i {
                Some(t.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let other = match block.2.as_ref() {
        Body::Block(b) => get_input_tensors(b),
        Body::Op(_) => vec![],
        Body::Sum(b) => get_input_tensors(b),
    };
    [inps, other].concat()
}

// Is this correct? I feel like it isn't but it might be. It treats the full grid (grid + threadblock) as a strict heirarchy.
fn grid_setter_idx(grid: &[u32]) -> String {
    let mut running_prod = 1;
    let mut s = "".to_string();
    for (i, g) in grid.iter().enumerate().rev() {
        if *g != 1 {
            if !s.is_empty() {
                s.push_str(" + ");
            }
            s.push_str(grid_levels(i as u8));
            if running_prod != 1 {
                s.push_str(&format!(" * {running_prod}"));
            }
            running_prod *= *g;
        }
    }
    if s.is_empty() {
        "0".to_string()
    } else {
        s
    }
}

fn get_sum_size(block: &Block) -> i64 {
    block.0
        * match block.2.as_ref() {
            Body::Block(b) => get_sum_size(b),
            Body::Op(_) | Body::Sum(_) => 1,
        }
}
