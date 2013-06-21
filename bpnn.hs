-----------------------------------
-- * Multilayer perceptron
----------------------------------

import System.Random
import Data.List
import Control.Monad.State
import Control.Monad.Writer
import Control.Arrow
import Debug.Trace

data NeuralNetwork = NeuralNetwork Nodes Nodes Nodes deriving (Show)

data Node = Node Activation Weights Momentums -- TODO make record syntax for automatic getter
instance Show Node where
  show (Node a w m) = "Node {\n\tActivation : " ++ show a ++ "\n\tWeights : " ++ show w ++ "\n\tMomemtums : " ++ show m ++ "\n}\n"
type Nodes = [Node]
type Activation = Double
type Weight = Double
type Weights = [Weight]
type Momentum = Double
type Momentums = [Momentum]

type PatternInput = [Double]
type PatternOutput = [Double]
type Pattern = (PatternInput,PatternOutput)
type Patterns = [Pattern]

type RandomState a = State StdGen a
type NeuralNetworkState = (NeuralNetwork, Double)

-----------------------------------
-- * Field getters
----------------------------------
getActivation :: Node -> Activation
getActivation (Node a _ _) = a

getWeights :: Node -> Weights
getWeights (Node _ w _) = w

getMomemtums :: Node -> Momentums
getMomemtums (Node _ _ m) = m

-----------------------------------
-- * Main
----------------------------------

main :: IO ()
main = do
  gen <- newStdGen
  let nn = initNetwork 2 2 1 gen
  let xor_patterns = [([0,0],[0])
                     ,([0,1],[1])
                     ,([1,0],[1])
                     ,([1,1],[0])]
  let (nn', e) = execState (trainFor xor_patterns 0.5 0.1 1000) (nn, 0)
  let results = test nn' xor_patterns
  mapM_ print results

----------------------------------
-- * Test
----------------------------------

test :: NeuralNetwork -> Patterns -> [String]
test nn [] = []
test nn ps = fmap (testPattern nn) ps

testPattern :: NeuralNetwork -> Pattern -> String
testPattern nn p =
  "Input : " ++ show i ++ " -> " ++ show (map getActivation outputNodes)
    where (i, o) = p
          (nn', _) = execState (update i) (nn, 0)
          (NeuralNetwork _ _ outputNodes) = nn'
----------------------------------
-- * Update
----------------------------------

trainFor :: Patterns -> Double -> Double -> Int -> State NeuralNetworkState ()
trainFor pats learn_rate delta_p iterations = do
  (nn, e) <- get
  let (nn', e') = last $ take iterations $ iterateS (train pats learn_rate delta_p) (nn, e)
  put (nn', e')
  return ()

train :: Patterns -> Double -> Double -> State NeuralNetworkState ()
train [] _ _ = return ()
train ps learn_rate delta_p = do
  (nn, e) <- get
  let (nn', e') = foldl (\acc x -> execState (trainPattern x learn_rate delta_p) acc) (nn, e) ps
  put (nn', e')
  return ()

trainPattern :: Pattern -> Double -> Double -> State NeuralNetworkState ()
trainPattern p learn_rate delta_p = do
  (nn, e) <- get
  let (nn_u, _ ) = execState (update $ fst p) (nn, e)
  let (nn' , e') = execState (backPropagate (snd p) learn_rate delta_p) (nn_u, e)
  trace (show nn') $ put (nn' , e')
  return ()

update :: PatternInput -> State NeuralNetworkState ()
update inputs = do
  (nn, e) <- get
  let (NeuralNetwork inputNodes hiddenNodes outputNodes) = nn
  let inputNodes' = applyInput inputs inputNodes
  let hiddenActivations = generateNextActivations inputNodes
  let hiddenNodes' = fmap(\(Node _ w m, a) -> Node a w m) $ zip hiddenNodes hiddenActivations
  let outputActivations = generateNextActivations hiddenNodes
  let outputNodes' = fmap(\(Node _ w m, a) -> Node a w m) $ zip outputNodes outputActivations
  let nn' = NeuralNetwork inputNodes hiddenNodes' outputNodes'
  put (nn', e)
  return ()

applyInput :: PatternInput -> Nodes -> Nodes
applyInput inputs inputNodes =
  inputNodes' ++ [biasNode]
    where biasNode = last inputNodes
          inputNodes' = fmap(\(Node a w m, a') -> (Node a' w m)) $ zip inputNodes inputs

generateNextActivations :: Nodes -> [Activation]
generateNextActivations = fmap generateNextActivation

generateNextActivation :: Node -> Activation
generateNextActivation (Node a w _) = sigmoid $ sum $ fmap (*a) w

backPropagate :: PatternOutput -> Double -> Double -> State NeuralNetworkState ()
backPropagate outputs learn_rate delta_p = do
  (nn, e) <- get
  let (NeuralNetwork inputNodes hiddenNodes outputNodes) = nn
  let output_errors = zipWith (-) (map getActivation outputNodes) outputs
  let output_deltas = (map dsigmoid >>> zipWith(*)) (map getActivation outputNodes) output_errors
  let hidden_errors = fmap (\(Node _ w _) -> sum $ zipWith (*) output_deltas w) hiddenNodes -- TODO
  let hidden_deltas = (map dsigmoid >>> zipWith(*)) (map getActivation hiddenNodes) hidden_errors
  let hiddenNodes'  = backPropagateResults output_deltas hiddenNodes learn_rate delta_p
  let inputNodes'   = backPropagateResults hidden_deltas inputNodes learn_rate delta_p
  let nn' = NeuralNetwork inputNodes' hiddenNodes' outputNodes
  let e' = foldl (\acc x -> acc + 0.5 * x ** 2) 0.0 $ zipWith (-) outputs (map getActivation outputNodes)
  put (nn', e')
  return ()

backPropagateResults :: [Double] -> Nodes -> Double -> Double -> Nodes
backPropagateResults deltas nodes learn_rate delta_p =
  fmap (\x -> backPropagateResult deltas x learn_rate delta_p) nodes

backPropagateResult :: [Double] -> Node -> Double -> Double -> Node
backPropagateResult deltas node learn_rate delta_p =
  Node a w' m'
    where (Node a w m) = node
          changes = fmap (* a) deltas
          -- curWeight + learnRate * change + momentum * curMomentum
          w' = (map (* delta_p) >>> zipWith(+)) m $ (map (* learn_rate) >>> zipWith(+)) changes w;
          m' = changes

----------------------------------
-- * Init
----------------------------------
initNetwork :: Int -> Int -> Int -> StdGen -> NeuralNetwork
initNetwork i' h o gen = do
  let i = i' + 1 -- input + 1 (bias node)
  let (iLayer, gen') = runState (initNodeLayer i h (-0.2, 0.2)) gen
  let (hLayer, _)    = runState (initNodeLayer h o (-2.0, 2.0)) gen'
  let (oLayer, _)    = runState (initNodeLayer o 0 (   0,   0)) gen'
  NeuralNetwork iLayer hLayer oLayer

initNodeLayer :: Int -> Int -> (Double, Double) -> RandomState Nodes
initNodeLayer ni no bnds = mapM (\x -> initNode no bnds) [1..ni]

initNode :: Int -> (Double, Double) -> RandomState Node
initNode n bnds = do
  let a = 1.0
  w <- randList bnds n
  let m = replicate n 0.0
  return (Node a w m)

----------------------------------
-- * Util
----------------------------------

getBoundedRandom :: Random a => (a,a) -> RandomState a
getBoundedRandom bnds = do
  gen <- get
  let (val, gen') = randomR bnds gen
  put gen'
  return val

runBoundedRandom :: Random a => (a,a) -> RandomState a
runBoundedRandom bnds = do
  oldState <- get
  let (result, newState) = runState (getBoundedRandom bnds) oldState
  put newState
  return result

-- Why can't this be mapM(const runBoundedRandom bnds) ?
randList :: Random a => (a,a) -> Int -> RandomState [a]
randList bnds n = mapM (\x -> runBoundedRandom bnds) [1..n]

iterateS :: State s a -> s -> [s]
iterateS f s0 =
  let x = execState f s0 in x : iterateS f x

sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

dsigmoid :: Double -> Double
dsigmoid x = 1.0 - x ** 2.0
