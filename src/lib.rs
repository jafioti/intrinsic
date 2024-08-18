use generational_box::{AnyStorage, GenerationalBox, Owner, UnsyncStorage};
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::hash::Hash;
use std::ops::*;

pub type FastHashMap<K, V> = FxHashMap<K, V>;

thread_local! {
    static EXPRESSION_OWNER: RefCell<Option<Owner<UnsyncStorage>>> = RefCell::new(Some(UnsyncStorage::owner()));
}

/// Clean up symbolic expresion storage
pub fn expression_cleanup() {
    EXPRESSION_OWNER.with(|cell| cell.borrow_mut().take());
}

/// Get the thread-local owner of expression storage
fn expression_owner() -> Owner {
    EXPRESSION_OWNER.with(|cell| cell.borrow().clone().unwrap())
}

#[derive(Clone, Copy)]
pub struct Expression {
    pub terms: GenerationalBox<Vec<Term>>,
}

impl Expression {
    fn new(terms: Vec<Term>) -> Self {
        Self {
            terms: expression_owner().insert(terms),
        }
    }
}

impl Hash for Expression {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.terms.read().hash(state);
    }
}

impl Default for Expression {
    fn default() -> Self {
        Expression::new(vec![])
    }
}

/// A single term of a symbolic expression such as a variable, number or operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Term {
    Num(i32),
    Var(char),
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    And,
    Or,
    Gte,
    Lt,
}

impl std::fmt::Debug for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Num(n) => write!(f, "{n}"),
            Term::Var(c) => write!(f, "{c}"),
            Term::Add => write!(f, "+"),
            Term::Sub => write!(f, "-"),
            Term::Mul => write!(f, "*"),
            Term::Div => write!(f, "/"),
            Term::Mod => write!(f, "%"),
            Term::Min => write!(f, "min"),
            Term::Max => write!(f, "max"),
            Term::And => write!(f, "&&"),
            Term::Or => write!(f, "||"),
            Term::Gte => write!(f, ">="),
            Term::Lt => write!(f, "<"),
        }
    }
}

impl Term {
    pub fn as_op(self) -> Option<fn(i64, i64) -> Option<i64>> {
        match self {
            Term::Add => Some(|a, b| a.checked_add(b)),
            Term::Sub => Some(|a, b| a.checked_sub(b)),
            Term::Mul => Some(|a, b| a.checked_mul(b)),
            Term::Div => Some(|a, b| a.checked_div(b)),
            Term::Mod => Some(|a, b| a.checked_rem(b)),
            Term::Max => Some(|a, b| Some(a.max(b))),
            Term::Min => Some(|a, b| Some(a.min(b))),
            Term::And => Some(|a, b| Some((a != 0 && b != 0) as i64)),
            Term::Or => Some(|a, b| Some((a != 0 || b != 0) as i64)),
            Term::Gte => Some(|a, b| Some((a >= b) as i64)),
            Term::Lt => Some(|a, b| Some((a < b) as i64)),
            _ => None,
        }
    }
}

impl Default for Term {
    fn default() -> Self {
        Self::Num(0)
    }
}

impl Expression {
    pub fn exec_stack(
        &self,
        variables: &FastHashMap<char, usize>,
        stack: &mut Vec<i64>,
    ) -> Option<usize> {
        for term in self.terms.read().iter() {
            match term {
                Term::Num(n) => stack.push(*n as i64),
                Term::Var(c) =>
                {
                    #[allow(clippy::needless_borrow)]
                    if let Some(n) = variables.get(&c) {
                        stack.push(*n as i64)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b).unwrap());
                }
            }
        }
        stack.pop().map(|i| i as usize)
    }
}

impl From<Term> for Expression {
    fn from(value: Term) -> Self {
        Expression::new(vec![value])
    }
}

impl From<char> for Expression {
    fn from(value: char) -> Self {
        Expression::new(vec![Term::Var(value)])
    }
}

impl From<&char> for Expression {
    fn from(value: &char) -> Self {
        Expression::new(vec![Term::Var(*value)])
    }
}

impl From<usize> for Expression {
    fn from(value: usize) -> Self {
        Expression::new(vec![Term::Num(value as i32)])
    }
}

impl From<&usize> for Expression {
    fn from(value: &usize) -> Self {
        Expression::new(vec![Term::Num(*value as i32)])
    }
}

impl From<i32> for Expression {
    fn from(value: i32) -> Self {
        Expression::new(vec![Term::Num(value)])
    }
}

impl From<&i32> for Expression {
    fn from(value: &i32) -> Self {
        Expression::new(vec![Term::Num(*value)])
    }
}

impl From<bool> for Expression {
    fn from(value: bool) -> Self {
        Expression::new(vec![Term::Num(value as i32)])
    }
}

impl From<&bool> for Expression {
    fn from(value: &bool) -> Self {
        Expression::new(vec![Term::Num(*value as i32)])
    }
}

impl Add<Expression> for usize {
    type Output = Expression;
    fn add(self, rhs: Expression) -> Self::Output {
        rhs + self
    }
}

impl Sub<Expression> for usize {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl Mul<Expression> for usize {
    type Output = Expression;
    fn mul(self, rhs: Expression) -> Self::Output {
        rhs * self
    }
}

impl Div<Expression> for usize {
    type Output = Expression;
    fn div(self, rhs: Expression) -> Self::Output {
        Expression::from(self) / rhs
    }
}

impl Rem<Expression> for usize {
    type Output = Expression;
    fn rem(self, rhs: Expression) -> Self::Output {
        Expression::from(self) % rhs
    }
}

impl BitAnd<Expression> for usize {
    type Output = Expression;
    fn bitand(self, rhs: Expression) -> Self::Output {
        rhs & self
    }
}

impl BitOr<Expression> for usize {
    type Output = Expression;
    fn bitor(self, rhs: Expression) -> Self::Output {
        rhs | self
    }
}

impl Add<Expression> for i32 {
    type Output = Expression;
    fn add(self, rhs: Expression) -> Self::Output {
        rhs + self
    }
}

impl Sub<Expression> for i32 {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl Mul<Expression> for i32 {
    type Output = Expression;
    fn mul(self, rhs: Expression) -> Self::Output {
        rhs * self
    }
}

impl Div<Expression> for i32 {
    type Output = Expression;
    fn div(self, rhs: Expression) -> Self::Output {
        Expression::from(self) / rhs
    }
}

impl Rem<Expression> for i32 {
    type Output = Expression;
    fn rem(self, rhs: Expression) -> Self::Output {
        Expression::from(self) % rhs
    }
}

impl BitAnd<Expression> for i32 {
    type Output = Expression;
    fn bitand(self, rhs: Expression) -> Self::Output {
        rhs & self
    }
}

impl BitOr<Expression> for i32 {
    type Output = Expression;
    fn bitor(self, rhs: Expression) -> Self::Output {
        rhs | self
    }
}

impl<E: Into<Expression>> Add<E> for Expression {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Add);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Sub<E> for Expression {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Sub);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Mul<E> for Expression {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Mul);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Div<E> for Expression {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Div);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Rem<E> for Expression {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Mod);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> BitAnd<E> for Expression {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::And);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> BitOr<E> for Expression {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Or);
        Expression::new(terms)
    }
}

impl std::iter::Product for Expression {
    fn product<I: Iterator<Item = Expression>>(mut iter: I) -> Self {
        let Some(mut p) = iter.next() else {
            return 0.into();
        };
        for n in iter {
            p *= n;
        }
        p
    }
}

impl<E: Into<Expression>> AddAssign<E> for Expression {
    fn add_assign(&mut self, rhs: E) {
        *self = *self + rhs;
    }
}

impl<E: Into<Expression>> SubAssign<E> for Expression {
    fn sub_assign(&mut self, rhs: E) {
        *self = *self - rhs;
    }
}

impl<E: Into<Expression>> MulAssign<E> for Expression {
    fn mul_assign(&mut self, rhs: E) {
        *self = *self * rhs;
    }
}

impl<E: Into<Expression>> DivAssign<E> for Expression {
    fn div_assign(&mut self, rhs: E) {
        *self = *self / rhs;
    }
}

impl<E: Into<Expression>> RemAssign<E> for Expression {
    fn rem_assign(&mut self, rhs: E) {
        *self = *self % rhs;
    }
}

impl<E: Into<Expression>> BitAndAssign<E> for Expression {
    fn bitand_assign(&mut self, rhs: E) {
        *self = *self & rhs;
    }
}

impl<E: Into<Expression>> BitOrAssign<E> for Expression {
    fn bitor_assign(&mut self, rhs: E) {
        *self = *self | rhs;
    }
}

#[cfg(test)]
mod tests {
    use crate::{Expression, FastHashMap};

    #[test]
    fn test() {
        let a = Expression::from('a');
        let b = Expression::from('b');
        let c: Expression = (100 + a * 2) / b;
        let mut vars = FastHashMap::default();
        vars.insert('a', 3);
        vars.insert('b', 2);
        assert_eq!(c.exec_stack(&vars, &mut Vec::new()).unwrap(), 53);
    }
}
