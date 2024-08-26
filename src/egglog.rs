// Wrapper around EGraph type

use egglog::ast::Command;
use egglog::Error;
use std::path::PathBuf;

use crate::{Expression, Term};

/// EGraph()
/// --
///
/// Create an empty EGraph.
pub struct EGraph {
    pub(crate) egraph: egglog::EGraph,
    cmds: Option<String>,
    pub(crate) expr_count: usize,
}

// type EggResult<T> = Result<T, Error>;

impl EGraph {
    fn new(
        fact_directory: Option<PathBuf>,
        seminaive: bool,
        terms_encoding: bool,
        record: bool,
    ) -> Self {
        let mut egraph = egglog::EGraph::default();
        egraph.fact_directory = fact_directory;
        egraph.seminaive = seminaive;
        if terms_encoding {
            egraph.enable_terms_encoding();
        }
        Self {
            egraph,
            expr_count: 0,
            cmds: if record { Some(String::new()) } else { None },
        }
    }

    /// Parse a program into a list of commands.
    fn parse_program(&self, input: &str) -> Option<Vec<Command>> {
        let commands = self.egraph.parse_program(input).unwrap();
        Some(commands)
    }

    /// Run a series of commands on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    fn run_program(&mut self, commands: Vec<Command>) -> Result<Vec<String>, Error> {
        let mut cmds_str = String::new();
        for cmd in &commands {
            cmds_str = cmds_str + &cmd.to_string() + "\n";
        }
        if let Some(cmds) = &mut self.cmds {
            cmds.push_str(&cmds_str);
        }

        self.egraph.run_program(commands)
    }

    // /// Returns the text of the commands that have been run so far, if `record` was passed.
    // fn commands(&self) -> Option<String> {
    //     self.cmds.clone()
    // }

    // /// Gets the last expressions extracted from the EGraph, if the last command
    // /// was a Simplify or Extract command.
    // fn extract_report(&mut self) -> Option<ExtractReport> {
    //     self.egraph.get_extract_report().as_ref().cloned()
    // }

    // /// Gets the last run report from the EGraph, if the last command
    // /// was a run or simplify command.
    // fn run_report(&mut self) -> Option<RunReport> {
    //     self.egraph.get_run_report().as_ref().cloned()
    // }

    // fn eval_i64(&mut self, expr: Expr) -> EggResult<i64> {
    //     self.eval_sort(expr, Arc::new(I64Sort::new("i64".into())))
    // }

    // fn eval_f64(&mut self, expr: Expr) -> EggResult<f64> {
    //     self.eval_sort(expr, Arc::new(F64Sort::new("f64".into())))
    // }

    // fn eval_string(&mut self, expr: Expr) -> EggResult<String> {
    //     let s: egglog::ast::Symbol =
    //         self.eval_sort(expr, Arc::new(StringSort::new("String".into())))?;
    //     Ok(s.to_string())
    // }

    // fn eval_bool(&mut self, expr: Expr) -> EggResult<bool> {
    //     self.eval_sort(expr, Arc::new(BoolSort::new("bool".into())))
    // }
}

// impl EGraph {
//     fn eval_sort<T: egglog::sort::Sort, V: egglog::sort::FromSort<Sort = T>>(
//         &mut self,
//         expr: Expr,
//         arcsort: Arc<T>,
//     ) -> EggResult<V> {
//         let expr: egglog::ast::Expr = expr.into();
//         let (_, value) = self.egraph.eval_expr(&expr)?;
//         Ok(V::load(&arcsort, &value))
//     }
// }

pub fn convert_expr_to_string(expr: crate::Expression) -> String {
    let mut stack = Vec::new();

    for term in expr.terms.read().iter() {
        match term {
            Term::Var(v) => {
                stack.push(format!("(Var \"{v}\")"));
            }
            Term::Num(n) => {
                stack.push(format!("(Num {n})"));
            }
            _ => {
                let left = stack.pop().unwrap();
                let right = stack.pop().unwrap();
                let term = term.term_name();
                stack.push(format!("({term} {left} {right})"));
            }
        }
    }
    stack.pop().unwrap()
}

pub fn convert_string_to_expr(input: &str) -> crate::Expression {
    let mut stack = vec![];
    for token in input
        .split(|c| c == ' ' || c == '(' || c == ')')
        .filter(|s| !s.trim().is_empty() && s.trim() != "Var" && s.trim() != "Num")
        .map(|s| s.trim_matches('"'))
        .rev()
    {
        if let Ok(n) = token.parse::<i32>() {
            stack.push(vec![Term::Num(n)]);
        } else if token.len() == 1 {
            stack.push(vec![Term::Var(token.chars().next().unwrap())]);
        } else {
            let b = stack.pop().unwrap();
            let mut a = stack.pop().unwrap();
            let op = match token {
                "Add" => Term::Add,
                "Sub" => Term::Sub,
                "Mul" => Term::Mul,
                "Div" => Term::Div,
                "Mod" => Term::Mod,
                "Min" => Term::Min,
                "Max" => Term::Max,
                "And" => Term::And,
                "Or" => Term::Or,
                "Gte" => Term::Gte,
                "Lt" => Term::Lt,
                _ => panic!("wtf is this: {token}"),
            };
            a.extend(b.into_iter());
            a.push(op);
            stack.push(a);
        }
    }

    Expression::new(stack.pop().unwrap())
}

const EGGLOG_VOCAB: &str = "
(datatype Math
	(Num i64)
	(Var String)
	(Add Math Math)
	(Sub Math Math)
	(Mul Math Math)
	(Div Math Math)
	(Mod Math Math)
	(Min Math Math)
	(Max Math Math)
	(And Math Math)
	(Or Math Math)
	(Gte Math Math)
	(Lt Math Math)
)";

const EGGLOG_RULES: &str = "
; Communative
(rewrite (Add a b) (Add b a))
(rewrite (Mul a b) (Mul b a))
(rewrite (Min a b) (Min b a))
(rewrite (Max a b) (Max b a))
(rewrite (And a b) (And b a))
(rewrite (Or a b) (Or b a))

; Associative
(rewrite (Add (Add a b) c) (Add a (Add b c)))
(rewrite (Mul (Mul a b) c) (Mul a (Mul b c)))
(rewrite (Div (Div a b) c) (Div a (Mul b c)))
(rewrite (Div (Mul a b) c) (Mul a (Div b c)))
(rewrite (Mul a (Div b c)) (Div (Mul a b) c))

; Distributive
(rewrite (Mul a (Add b c)) (Add (Mul a b) (Mul a c)))
(rewrite (Div (Add a b) c) (Add (Div a c) (Div b c)))

; Constant folding
(rewrite (Add (Num a) (Num b)) (Num (+ a b)))
(rewrite (Sub (Num a) (Num b)) (Num (- a b)))
(rewrite (Mul (Num a) (Num b)) (Num (* a b)))
(rewrite (Div (Num a) (Num b)) (Num (/ a b)) :when ((!= 0 b) (= 0 (% a b))))
(rewrite (Max (Num a) (Num b)) (Num (max a b)))
(rewrite (Min (Num a) (Num b)) (Num (min a b)))
(rewrite (And (Num a) (Num b)) (Num (& a b)))
(rewrite (Or (Num a) (Num b)) (Num (| a b)))

; Factoring
(rewrite (Add (Mul a b) (Mul a c)) (Mul a (Add b c)))
(rewrite (Add a a) (Mul (Num 2) a))

; Other
(rewrite (Add (Div a b) c) (Div (Add a (Mul c b)) b))
";

pub fn create_egraph() -> EGraph {
    let mut egraph = EGraph::new(None, false, false, false);
    let commands = egraph
        .parse_program(&format!(
            "{EGGLOG_VOCAB}
{EGGLOG_RULES}",
        ))
        .unwrap();
    egraph.run_program(commands).unwrap();
    egraph
}

pub fn egglog_simplify_egraph(expr: Expression, egraph: &mut EGraph, iters: usize) -> Expression {
    let expr_string = convert_expr_to_string(expr);
    let commands = egraph
        .parse_program(&format!(
            "
(let expr{} {expr_string})
;(run-schedule (saturate (run)))
(run {iters})
(extract expr{})",
            egraph.expr_count, egraph.expr_count
        ))
        .unwrap();
    egraph.expr_count += 1;
    let result = egraph.run_program(commands).unwrap();
    convert_string_to_expr(&result[0])
}

pub fn egglog_simplify(mut expr: Expression, inner: usize, outer: usize) -> Expression {
    for _ in 0..outer {
        expr = egglog_simplify_egraph(expr, &mut create_egraph(), inner);
    }
    expr
}

pub fn check_equals(a: Expression, b: Expression) -> bool {
    let mut egraph = create_egraph();
    let a_string = convert_expr_to_string(a);
    let b_string = convert_expr_to_string(b);
    let commands = egraph
        .parse_program(&format!(
            "
(let expr1 {a_string})
(let expr2 {b_string})
(run 5)
(check (= expr1 expr2))",
        ))
        .unwrap();
    egraph.run_program(commands).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{expr, expression_cleanup, Expression};

    #[test]
    fn test_simplify() {
        let expr = (Expression::from('x') + 2 + 8) * (4 * 3) + 14;
        assert!(check_equals(
            Expression::from('x') * 12 + 134,
            egglog_simplify(expr, 5, 5)
        ));
        expression_cleanup();
    }

    #[test]
    fn test_simplify_hard() {
        let o = (expr('z')
            / ((-5
                + (((((-5 + ((((((expr('w') + 153) / 2) / 2) / 2) / 2) / 2)) * 4) + 9) / 2) / 2))
                * (-5
                    + (((9 + (4 * (-5 + ((((((153 + expr('h')) / 2) / 2) / 2) / 2) / 2)))) / 2)
                        / 2))))
            % 64;
        let r = egglog_simplify(o, 5, 5);
        assert!(r.len() <= 13);
        assert!(check_equals(
            r,
            (expr('z') / (((expr('w') + -95) * (expr('h') + -95)) / 1024)) % 64
        ));
        expression_cleanup();
    }
}
