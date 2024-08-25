use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

use mcts::transposition_table::*;
use mcts::tree_policy::*;
use mcts::*;

const TARGET: &str = "How are you doing today?";

#[derive(Clone)]
struct Game(String);

impl GameState for Game {
    type Move = char;
    type Player = ();
    type MoveList = Vec<Self::Move>;

    fn current_player(&self) -> Self::Player {}

    fn available_moves(&self) -> Vec<Self::Move> {
        if self.0.len() >= TARGET.len() {
            vec![]
        } else {
            (0..255).map(|i| i as u8 as char).collect()
        }
    }

    fn make_move(&mut self, mov: &Self::Move) {
        self.0.push(*mov);
    }
}

impl TranspositionHash for Game {
    fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
}

struct MyEvaluator;

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &Game,
        moves: &Vec<char>,
        _: Option<SearchHandle<MyMCTS>>,
    ) -> (Vec<()>, i64) {
        let reward = TARGET
            .chars()
            .take(state.0.len())
            .zip(state.0.chars())
            .map(|(a, b)| (a == b) as i64)
            .sum();
        (vec![(); moves.len()], reward)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(&self, _: &Game, evaln: &i64, _: SearchHandle<MyMCTS>) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct MyMCTS;

impl MCTS for MyMCTS {
    type State = Game;
    type Eval = MyEvaluator;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn virtual_loss(&self) -> i64 {
        0
    }

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mcts_test() {
        let mut mcts = MCTSManager::new(
            Game("".to_string()),
            MyMCTS,
            MyEvaluator,
            UCTPolicy::new(0.01),
            ApproxTable::new(1024),
        );
        mcts.playout_parallel_for(std::time::Duration::from_millis(500), 16);
        let pv: Vec<_> = mcts
            .principal_variation_states(100)
            .into_iter()
            .map(|x| x.0)
            .collect();
        assert_eq!(pv.last().unwrap(), TARGET);
    }
}
