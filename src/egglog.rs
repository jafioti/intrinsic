// Wrapper around EGraph type

use egglog::ast::{Command, Expr};
use egglog::sort::{BoolSort, F64Sort, I64Sort, StringSort};
use egglog::{Error, ExtractReport, RunReport};
use std::path::PathBuf;
use std::sync::Arc;

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
    fn parse_program(&mut self, input: &str) -> Option<Vec<Command>> {
        let commands = self.egraph.parse_program(input).unwrap();
        Some(commands.into_iter().map(|x| x.into()).collect())
    }

    /// Run a series of commands on the EGraph.
    /// Returns a list of strings representing the output.
    /// An EggSmolError is raised if there is problem parsing or executing.
    fn run_program(&mut self, commands: Vec<Command>) -> Result<Vec<String>, Error> {
        let commands: Vec<egglog::ast::Command> = commands.into_iter().map(|x| x.into()).collect();
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
