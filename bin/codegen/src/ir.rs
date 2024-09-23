#[derive(Debug)]
pub struct Ref(pub i64);
#[derive(Debug)]
pub struct Tensor(pub String);
#[derive(Debug)]
pub enum Input {
    Block(Block),
    Ref(Ref),
    Tensor(Tensor),
}
impl From<Block> for Input {
    fn from(value: Block) -> Self {
        Input::Block(value)
    }
}
impl From<Ref> for Input {
    fn from(value: Ref) -> Self {
        Input::Ref(value)
    }
}
impl From<Tensor> for Input {
    fn from(value: Tensor) -> Self {
        Input::Tensor(value)
    }
}

pub trait ToInputs {
    fn to_inputs(self) -> Vec<(Input, i64)>;
}
impl<T: Into<Input>> ToInputs for (T, i64) {
    fn to_inputs(self) -> Vec<(Input, i64)> {
        vec![(self.0.into(), self.1)]
    }
}
impl<T: Into<Input>> ToInputs for ((T, i64),) {
    fn to_inputs(self) -> Vec<(Input, i64)> {
        vec![(self.0 .0.into(), self.0 .1)]
    }
}
impl<T: Into<Input>, S: Into<Input>> ToInputs for ((T, i64), (S, i64)) {
    fn to_inputs(self) -> Vec<(Input, i64)> {
        vec![(self.0 .0.into(), self.0 .1), (self.1 .0.into(), self.1 .1)]
    }
}

#[derive(Debug)]
pub enum Op {
    Ref(i32),
    Add(Box<Op>, Box<Op>),
    Mul(Box<Op>, Box<Op>),
    Exp(Box<Op>),
    Sin(Box<Op>),
}
pub fn add(a: Op, b: Op) -> Op {
    Op::Add(Box::new(a), Box::new(b))
}
pub fn mul(a: Op, b: Op) -> Op {
    Op::Mul(Box::new(a), Box::new(b))
}
pub fn exp(a: Op) -> Op {
    Op::Exp(Box::new(a))
}
pub fn sin(a: Op) -> Op {
    Op::Sin(Box::new(a))
}
pub fn refr(i: i32) -> Op {
    Op::Ref(i)
}

#[derive(Debug)]
pub struct Block(pub i64, pub Vec<(Input, i64)>, pub Box<Body>);

#[derive(Debug)]
pub enum Body {
    Block(Block),
    Sum(Block),
    Op(Op),
}

impl From<Block> for Body {
    fn from(value: Block) -> Self {
        Body::Block(value)
    }
}
impl From<Op> for Body {
    fn from(value: Op) -> Self {
        Body::Op(value)
    }
}

pub fn block(size: i64, inps: impl ToInputs, body: impl Into<Body>) -> Block {
    Block(size, inps.to_inputs(), Box::new(body.into()))
}

pub fn sum(block: Block) -> Body {
    Body::Sum(block)
}
