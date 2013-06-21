-----------------------------------
-- * Multilayer perceptron
----------------------------------

import System.Random
import Data.List
import Control.Monad.State
import Control.Monad.Writer
import Control.Arrow
import Control.Applicative ((<$>))
import Debug.Trace

data NeuralNetwork = NeuralNetwork Nodes Nodes Nodes deriving Show

data Node = Node { activation :: Activation
                 , weights    :: Weights
                 , momentums  :: Momentums
                 } deriving Show

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
  let (nn', _) = execState (trainFor xor_patterns 0.5 0.1 1000) (nn, 0)
  let results = test nn' xor_patterns
  mapM_ print results

----------------------------------
-- * Test
----------------------------------

test :: NeuralNetwork -> Patterns -> [String]
test _ [] = []
test nn ps = testPattern nn <$> ps

testPattern :: NeuralNetwork -> Pattern -> String
testPattern nn (i, _) =
  "Input : " ++ show i ++ " -> " ++ show (map activation outputNodes)
    where (NeuralNetwork _ _ outputNodes, _) = execState (update i) (nn, 0)

----------------------------------
-- * Update
----------------------------------

trainFor :: Patterns -> Double -> Double -> Int -> State NeuralNetworkState ()
trainFor pats learn_rate delta_p iterations =
  get >>= put . last . take iterations . iterateS (train pats learn_rate delta_p)

train :: Patterns -> Double -> Double -> State NeuralNetworkState ()
train [] _ _ = return ()
train ps learn_rate delta_p = do
  t <- get
  put $ foldl (\acc x -> execState (trainPattern x learn_rate delta_p) acc) t ps

trainPattern :: Pattern -> Double -> Double -> State NeuralNetworkState ()
trainPattern p learn_rate delta_p = do
  t@(nn, e) <- get
  let (nn_u, _) = execState (update $ fst p) t
  let (nn' , e') = execState (backPropagate (snd p) learn_rate delta_p) (nn_u, e)
  traceShow nn' $ put (nn' , e') -- Stop.

update :: PatternInput -> State NeuralNetworkState ()
update inputs = do
  (NeuralNetwork inputNodes hiddenNodes outputNodes, e) <- get
  let inputNodes' = applyInput inputs inputNodes
  let hiddenActivations = generateNextActivations inputNodes
  let hiddenNodes' = zwa hiddenNodes hiddenActivations
  let outputActivations = generateNextActivations hiddenNodes
  let outputNodes' = zwa outputNodes outputActivations
  put (NeuralNetwork inputNodes hiddenNodes' outputNodes', e)
  where zwa = zipWith (\n a -> n { activation = a })

applyInput :: PatternInput -> Nodes -> Nodes
applyInput inputs inputNodes =
  inputNodes' ++ [last inputNodes]
    where inputNodes' = zipWith (\n a' -> n { activation = a' }) inputNodes inputs

generateNextActivations :: Nodes -> [Activation]
generateNextActivations = fmap generateNextActivation

generateNextActivation :: Node -> Activation
generateNextActivation (Node a w _) = sigmoid . sum $ (* a) <$> w

backPropagate :: PatternOutput -> Double -> Double -> State NeuralNetworkState ()
backPropagate outputs learn_rate delta_p = do
  (NeuralNetwork inputNodes hiddenNodes outputNodes, _) <- get
  let output_errors = zipWith (-) (mapA outputNodes) outputs
  let output_deltas = mDSigAct outputNodes output_errors
  let hidden_errors = sum . zipWith (*) output_deltas . weights <$> hiddenNodes
  let hidden_deltas = mDSigAct hiddenNodes hidden_errors
  let hiddenNodes'  = propRes output_deltas hiddenNodes
  let inputNodes'   = propRes hidden_deltas inputNodes
  let e' = foldl (\acc x -> acc + 0.5 * x ** 2.0) 0.0 $ zipWith (-) outputs $ mapA outputNodes
  put (NeuralNetwork inputNodes' hiddenNodes' outputNodes, e')
  where propRes x y  = backPropagateResults x y learn_rate delta_p
        mapA         = map activation
        mDSigAct x   = map dsigmoid >>> zipWith (*) $ mapA x

backPropagateResults :: [Double] -> Nodes -> Double -> Double -> Nodes
backPropagateResults deltas nodes learn_rate delta_p =
  (\x -> backPropagateResult deltas x learn_rate delta_p) <$> nodes

backPropagateResult :: [Double] -> Node -> Double -> Double -> Node
backPropagateResult deltas (Node a w m) learn_rate delta_p =
  Node a w' changes
    where changes = (* a) <$> deltas
          -- curWeight + learnRate * change + momentum * curMomentum
          w' = mzw delta_p m $ mzw learn_rate changes w
          mzw x = map (* x) >>> zipWith (+)

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
initNodeLayer ni no bnds = mapM (const $ initNode no bnds) [1 .. ni]

initNode :: Int -> (Double, Double) -> RandomState Node
initNode n bnds = randList bnds n >>= \w -> return $ Node 1.0 w $ replicate n 0.0

----------------------------------
-- * Util
----------------------------------

getBoundedRandom :: Random a => (a,a) -> RandomState a
getBoundedRandom bnds = get >>= putRet . randomR bnds

runBoundedRandom :: Random a => (a,a) -> RandomState a
runBoundedRandom bnds = get >>= putRet . runState (getBoundedRandom bnds)

putRet :: MonadState s m => (b, s) -> m b
putRet (r, s) = put s >> return r

randList :: Random a => (a,a) -> Int -> RandomState [a]
randList bnds n = mapM (const $ runBoundedRandom bnds) [1..n]

iterateS :: State s a -> s -> [s]
iterateS f s0 =
  let x = execState f s0 in x : iterateS f x

sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

dsigmoid :: Double -> Double
dsigmoid x = 1.0 - x ** 2.0
