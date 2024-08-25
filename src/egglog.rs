// Wrapper around EGraph type

use egglog::ast::{Command, Expr};
use egglog::sort::{BoolSort, F64Sort, I64Sort, StringSort};
use egglog::{Error, ExtractReport, RunReport};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use crate::{Expression, Term};

/// EGraph()
/// --
///
/// Create an empty EGraph.
pub struct EGraph {
    pub(crate) egraph: egglog::EGraph,
    cmds: Option<String>,
}

type EggResult<T> = Result<T, Error>;

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

            cmds: if record { Some(String::new()) } else { None },
        }
    }

    /// Parse a program into a list of commands.
    fn parse_program(&self, input: &str) -> Option<Vec<Command>> {
        let commands = self.egraph.parse_program(input).unwrap();
        Some(commands.into_iter().map(|x| x.into()).collect())
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

    /// Returns the text of the commands that have been run so far, if `record` was passed.
    fn commands(&self) -> Option<String> {
        self.cmds.clone()
    }

    /// Gets the last expressions extracted from the EGraph, if the last command
    /// was a Simplify or Extract command.
    fn extract_report(&mut self) -> Option<ExtractReport> {
        self.egraph.get_extract_report().as_ref().cloned()
    }

    /// Gets the last run report from the EGraph, if the last command
    /// was a run or simplify command.
    fn run_report(&mut self) -> Option<RunReport> {
        self.egraph.get_run_report().as_ref().cloned()
    }

    fn eval_i64(&mut self, expr: Expr) -> EggResult<i64> {
        self.eval_sort(expr, Arc::new(I64Sort::new("i64".into())))
    }

    fn eval_f64(&mut self, expr: Expr) -> EggResult<f64> {
        self.eval_sort(expr, Arc::new(F64Sort::new("f64".into())))
    }

    fn eval_string(&mut self, expr: Expr) -> EggResult<String> {
        let s: egglog::ast::Symbol =
            self.eval_sort(expr, Arc::new(StringSort::new("String".into())))?;
        Ok(s.to_string())
    }

    fn eval_bool(&mut self, expr: Expr) -> EggResult<bool> {
        self.eval_sort(expr, Arc::new(BoolSort::new("bool".into())))
    }
}

impl EGraph {
    fn eval_sort<T: egglog::sort::Sort, V: egglog::sort::FromSort<Sort = T>>(
        &mut self,
        expr: Expr,
        arcsort: Arc<T>,
    ) -> EggResult<V> {
        let expr: egglog::ast::Expr = expr.into();
        let (_, value) = self.egraph.eval_expr(&expr)?;
        Ok(V::load(&arcsort, &value))
    }
}

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

pub fn convert_string_to_expr(input: &str) -> Result<crate::Expression, String> {
    let mut tokens: Vec<&str> = input
        .split(|c| c == ' ' || c == '(' || c == ')')
        .filter(|s| s.trim().len() > 0)
        .rev()
        .collect();
    let mut stack = VecDeque::new();
    let mut output = Vec::new();

    while let Some(token) = tokens.pop() {
        match token {
            "Add" => stack.push_back(Term::Add),
            "Sub" => stack.push_back(Term::Sub),
            "Mul" => stack.push_back(Term::Mul),
            "Div" => stack.push_back(Term::Div),
            "Mod" => stack.push_back(Term::Mod),
            "Min" => stack.push_back(Term::Min),
            "Max" => stack.push_back(Term::Max),
            "And" => stack.push_back(Term::And),
            "Or" => stack.push_back(Term::Or),
            "Gte" => stack.push_back(Term::Gte),
            "Lt" => stack.push_back(Term::Lt),
            "Var" => {
                if let Some(var_name) = tokens.pop() {
                    let var = var_name
                        .trim_matches('"')
                        .chars()
                        .next()
                        .ok_or("Empty variable name")?;
                    output.push(Term::Var(var));
                } else {
                    return Err("Expected variable name after 'Var'".to_string());
                }
            }
            "Num" => {
                if let Some(num_str) = tokens.pop() {
                    let num = num_str.parse::<i32>().map_err(|e| e.to_string())?;
                    output.push(Term::Num(num));
                } else {
                    return Err("Expected number after 'Num'".to_string());
                }
            }
            _ => return Err(format!("Unknown token: {}", token)),
        }

        if output.len() >= 2 && !stack.is_empty() {
            output.push(stack.pop_back().unwrap());
        }
    }

    while let Some(op) = stack.pop_back() {
        output.push(op);
    }

    Ok(crate::Expression::new(output))
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
(rewrite (Div (Div a b) c) (Div a (Mul b c))) ; Weird ones, probably not nessecary
(rewrite (Div (Mul a b) c) (Mul a (Div b c)))

; Distributive
(rewrite (Mul a (Add b c)) (Add (Mul a b) (Mul a c)))
(rewrite (Div (Add a b) c) (Add (Div a c) (Div b c)))

; Constant folding
(rewrite (Add (Num a) (Num b)) (Num (+ a b)))
(rewrite (Sub (Num a) (Num b)) (Num (- a b)))
(rewrite (Mul (Num a) (Num b)) (Num (* a b)))
";

pub fn egglog_simplify(expr: Expression) -> Expression {
    let mut egraph = EGraph::new(None, false, false, false);
    let expr_string = convert_expr_to_string(expr);
    let commands = egraph
        .parse_program(&format!(
            "{EGGLOG_VOCAB}
{EGGLOG_RULES}
(let expr1 {expr_string})
(run-schedule (saturate (run)))
(extract expr1)",
        ))
        .unwrap();
    let result = egraph.run_program(commands).unwrap();
    convert_string_to_expr(&result[0]).unwrap()
}

pub fn check_equals(a: Expression, b: Expression) -> bool {
    let mut egraph = EGraph::new(None, false, false, false);
    let a_string = convert_expr_to_string(a);
    let b_string = convert_expr_to_string(b);
    let commands = egraph
        .parse_program(&format!(
            "{EGGLOG_VOCAB}
{EGGLOG_RULES}
(let expr1 {a_string})
(let expr2 {b_string})
(run-schedule (saturate (run)))
(check (= expr1 expr2))",
        ))
        .unwrap();
    egraph.run_program(commands).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{expression_cleanup, Expression};

    #[test]
    fn test_simplify() {
        let expr = (Expression::from('x') + 2 + 8) * (4 * 3) + 14;
        assert!(check_equals(
            Expression::from('x') * 12 + 134,
            egglog_simplify(expr)
        ));
        expression_cleanup();
    }
}
