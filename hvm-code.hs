
-- BLOCK 0:
module HVML.Collapse where
-- BLOCK 1:
import Control.Monad (ap, forM, forM_)
import Control.Monad.IO.Class
import Data.Char (chr, ord)
import Data.IORef
import Data.Word
import GHC.Conc
import HVML.Show
import HVML.Type
import System.Exit (exitFailure)
import System.IO.Unsafe (unsafeInterleaveIO)
import qualified Data.IntMap.Strict as IM
import qualified Data.Map.Strict as MS
import Debug.Trace
-- BLOCK 2:
-- The Collapse Monad
-- ------------------
-- See: https://gist.github.com/VictorTaelin/60d3bc72fb4edefecd42095e44138b41
-- BLOCK 3:
-- A bit-string
data Bin
  = O Bin
  | I Bin
  | E
  deriving Show
-- BLOCK 4:
-- A Collapse is a tree of superposed values
data Collapse a = CSup Word64 (Collapse a) (Collapse a) | CVal a | CEra
  deriving Show
-- BLOCK 5:
bind :: Collapse a -> (a -> Collapse b) -> Collapse b
bind k f = fork k IM.empty where
  -- fork :: Collapse a -> IntMap (Bin -> Bin) -> Collapse b
  fork CEra         paths = CEra
  fork (CVal v)     paths = pass (f v) (IM.map (\x -> x E) paths)
  fork (CSup k x y) paths =
    let lft = fork x $ IM.alter (\x -> Just (maybe (putO id) putO x)) (fromIntegral k) paths in
    let rgt = fork y $ IM.alter (\x -> Just (maybe (putI id) putI x)) (fromIntegral k) paths in
    CSup k lft rgt 
  -- pass :: Collapse b -> IntMap Bin -> Collapse b
  pass CEra         paths = CEra
  pass (CVal v)     paths = CVal v
  pass (CSup k x y) paths = case IM.lookup (fromIntegral k) paths of
    Just (O p) -> pass x (IM.insert (fromIntegral k) p paths)
    Just (I p) -> pass y (IM.insert (fromIntegral k) p paths)
    Just E     -> CSup k x y
    Nothing    -> CSup k x y
  -- putO :: (Bin -> Bin) -> (Bin -> Bin)
  putO bs = \x -> bs (O x)
  -- putI :: (Bin -> Bin) -> (Bin -> Bin) 
  putI bs = \x -> bs (I x)
-- BLOCK 6:
-- Mutates an element at given index in a list
mut :: Word64 -> (a -> a) -> [a] -> [a]
mut 0 f (x:xs) = f x : xs
mut n f (x:xs) = x : mut (n-1) f xs
mut _ _ []     = []
-- BLOCK 7:
instance Functor Collapse where
  fmap f (CVal v)     = CVal (f v)
  fmap f (CSup k x y) = CSup k (fmap f x) (fmap f y)
  fmap _ CEra         = CEra
-- BLOCK 8:
instance Applicative Collapse where
  pure  = CVal
  (<*>) = ap
-- BLOCK 9:
instance Monad Collapse where
  return = pure
  (>>=)  = bind
-- BLOCK 10:
-- Dup Collapser
-- -------------
-- BLOCK 11:
collapseDupsAt :: IM.IntMap [Int] -> ReduceAt -> Book -> Loc -> HVM Core
-- BLOCK 12:
collapseDupsAt state@(paths) reduceAt book host = unsafeInterleaveIO $ do
  term <- reduceAt book host
  case tagT (termTag term) of
-- BLOCK 13:
    ERA -> do
      return Era
-- BLOCK 14:
    LET -> do
      let loc = termLoc term
      let mode = modeT (termLab term)
      name <- return $ "$" ++ show (loc + 0)
      val0 <- collapseDupsAt state reduceAt book (loc + 1)
      bod0 <- collapseDupsAt state reduceAt book (loc + 2)
      return $ Let mode name val0 bod0
-- BLOCK 15:
    LAM -> do
      let loc = termLoc term
      name <- return $ "$" ++ show (loc + 0)
      bod0 <- collapseDupsAt state reduceAt book (loc + 0)
      return $ Lam name bod0
-- BLOCK 16:
    APP -> do
      let loc = termLoc term
      fun0 <- collapseDupsAt state reduceAt book (loc + 0)
      arg0 <- collapseDupsAt state reduceAt book (loc + 1)
      return $ App fun0 arg0
-- BLOCK 17:
    SUP -> do
      let loc = termLoc term
      let lab = termLab term
      case IM.lookup (fromIntegral lab) paths of
        Just (p:ps) -> do
          let newPaths = IM.insert (fromIntegral lab) ps paths
          collapseDupsAt (newPaths) reduceAt book (loc + fromIntegral p)
        _ -> do
          tm00 <- collapseDupsAt state reduceAt book (loc + 0)
          tm11 <- collapseDupsAt state reduceAt book (loc + 1)
          return $ Sup lab tm00 tm11
-- BLOCK 18:
    VAR -> do
      let loc = termLoc term
      sub <- got loc
      if termGetBit sub /= 0
      then do
        set (loc + 0) (termRemBit sub)
        collapseDupsAt state reduceAt book (loc + 0)
      else do
        name <- return $ "$" ++ show loc
        return $ Var name
-- BLOCK 19:
    DP0 -> do
      let loc = termLoc term
      let lab = termLab term
      sb0 <- got (loc+0)
      if termGetBit sb0 /= 0
      then do
        set (loc + 0) (termRemBit sb0)
        collapseDupsAt state reduceAt book (loc + 0)
      else do
        let newPaths = IM.alter (Just . maybe [0] (0:)) (fromIntegral lab) paths
        collapseDupsAt (newPaths) reduceAt book (loc + 0)
-- BLOCK 20:
    DP1 -> do
      let loc = termLoc term
      let lab = termLab term
      sb1 <- got (loc+1)
      if termGetBit sb1 /= 0
      then do
        set (loc + 1) (termRemBit sb1)
        collapseDupsAt state reduceAt book (loc + 1)
      else do
        let newPaths = IM.alter (Just . maybe [1] (1:)) (fromIntegral lab) paths
        collapseDupsAt (newPaths) reduceAt book (loc + 0)
-- BLOCK 21:
    CTR -> do
      let loc = termLoc term
      let lab = termLab term
      let cid = u12v2X lab
      let ari = u12v2Y lab
      let aux = if ari == 0 then [] else [loc + i | i <- [0..ari-1]]
      fds0 <- forM aux (collapseDupsAt state reduceAt book)
      return $ Ctr cid fds0
-- BLOCK 22:
    MAT -> do
      let loc = termLoc term
      let len = u12v2X $ termLab term
      let aux = if len == 0 then [] else [loc + 1 + i | i <- [0..len-1]]
      val0 <- collapseDupsAt state reduceAt book (loc + 0)
      css0 <- forM aux $ \h -> do
        bod <- collapseDupsAt state reduceAt book h
        return $ ("#", [], bod) -- TODO: recover constructor and fields
      return $ Mat val0 [] css0
-- BLOCK 23:
    W32 -> do
      let val = termLoc term
      return $ U32 (fromIntegral val)
-- BLOCK 24:
    CHR -> do
      let val = termLoc term
      return $ Chr (chr (fromIntegral val))
-- BLOCK 25:
    OPX -> do
      let loc = termLoc term
      let opr = toEnum (fromIntegral (termLab term))
      nm00 <- collapseDupsAt state reduceAt book (loc + 0)
      nm10 <- collapseDupsAt state reduceAt book (loc + 1)
      return $ Op2 opr nm00 nm10
-- BLOCK 26:
    OPY -> do
      let loc = termLoc term
      let opr = toEnum (fromIntegral (termLab term))
      nm00 <- collapseDupsAt state reduceAt book (loc + 0)
      nm10 <- collapseDupsAt state reduceAt book (loc + 1)
      return $ Op2 opr nm00 nm10
-- BLOCK 27:
    REF -> do
      let loc = termLoc term
      let lab = termLab term
      let fid = u12v2X lab
      let ari = u12v2Y lab
      arg0 <- mapM (collapseDupsAt state reduceAt book) [loc + i | i <- [0..ari-1]]
      let name = MS.findWithDefault "?" fid (idToName book)
      return $ Ref name fid arg0
-- BLOCK 28:
    tag -> do
      putStrLn ("unexpected-tag:" ++ show tag)
      return $ Var "?"
      -- exitFailure
-- BLOCK 29:
-- Sup Collapser
-- -------------
-- BLOCK 30:
collapseSups :: Book -> Core -> Collapse Core
-- BLOCK 31:
collapseSups book core = case core of
-- BLOCK 32:
  Var name -> do
    return $ Var name
-- BLOCK 33:
  Ref name fid args -> do
    args <- mapM (collapseSups book) args
    return $ Ref name fid args
-- BLOCK 34:
  Lam name body -> do
    body <- collapseSups book body
    return $ Lam name body
-- BLOCK 35:
  App fun arg -> do
    fun <- collapseSups book fun
    arg <- collapseSups book arg
    return $ App fun arg
-- BLOCK 36:
  Dup lab x y val body -> do
    val <- collapseSups book val
    body <- collapseSups book body
    return $ Dup lab x y val body
-- BLOCK 37:
  Ctr cid fields -> do
    fields <- mapM (collapseSups book) fields
    return $ Ctr cid fields
-- BLOCK 38:
  Mat val mov css -> do
    val <- collapseSups book val
    mov <- mapM (\(key, expr) -> do
      expr <- collapseSups book expr
      return (key, expr)) mov
    css <- mapM (\(ctr, fds, bod) -> do
      bod <- collapseSups book bod
      return (ctr, fds, bod)) css
    return $ Mat val mov css
-- BLOCK 39:
  U32 val -> do
    return $ U32 val
-- BLOCK 40:
  Chr val -> do
    return $ Chr val
-- BLOCK 41:
  Op2 op x y -> do
    x <- collapseSups book x
    y <- collapseSups book y
    return $ Op2 op x y
-- BLOCK 42:
  Let mode name val body -> do
    val <- collapseSups book val
    body <- collapseSups book body
    return $ Let mode name val body
-- BLOCK 43:
  Era -> do
    CEra
-- BLOCK 44:
  Sup lab tm0 tm1 -> do
    let tm0' = collapseSups book tm0
    let tm1' = collapseSups book tm1
    CSup lab tm0' tm1'
-- BLOCK 45:
-- Tree Collapser
-- --------------
-- BLOCK 46:
doCollapseAt :: ReduceAt -> Book -> Loc -> HVM (Collapse Core)
doCollapseAt reduceAt book host = do
  -- namesRef <- newIORef MS.empty
  let state = (IM.empty)
  core <- collapseDupsAt state reduceAt book host
  return $ collapseSups book core
-- BLOCK 47:
-- Priority Queue
-- --------------
-- BLOCK 48:
data PQ a
  = PQLeaf
  | PQNode (Word64, a) (PQ a) (PQ a)
  deriving (Show)
-- BLOCK 49:
pqUnion :: PQ a -> PQ a -> PQ a
pqUnion PQLeaf heap = heap
pqUnion heap PQLeaf = heap
pqUnion heap1@(PQNode (k1,v1) l1 r1) heap2@(PQNode (k2,v2) l2 r2)
  | k1 <= k2  = PQNode (k1,v1) (pqUnion heap2 r1) l1
  | otherwise = PQNode (k2,v2) (pqUnion heap1 r2) l2
-- BLOCK 50:
pqPop :: PQ a -> Maybe ((Word64, a), PQ a)
pqPop PQLeaf         = Nothing
pqPop (PQNode x l r) = Just (x, pqUnion l r)
-- BLOCK 51:
pqPut :: (Word64,a) -> PQ a -> PQ a
pqPut (k,v) = pqUnion (PQNode (k,v) PQLeaf PQLeaf)
-- BLOCK 52:
-- Simple Queue
-- ------------
-- Allows pushing to an end, and popping from another.
-- Simple purely functional implementation.
-- Includes sqPop and sqPut.
-- BLOCK 53:
data SQ a = SQ [a] [a]
-- BLOCK 54:
sqPop :: SQ a -> Maybe (a, SQ a)
sqPop (SQ [] [])     = Nothing
sqPop (SQ [] ys)     = sqPop (SQ (reverse ys) [])
sqPop (SQ (x:xs) ys) = Just (x, SQ xs ys)
-- BLOCK 55:
sqPut :: a -> SQ a -> SQ a
sqPut x (SQ xs ys) = SQ xs (x:ys)
-- BLOCK 56:
-- Flattener
-- ---------
-- BLOCK 57:
flattenDFS :: Collapse a -> [a]
flattenDFS (CSup k a b) = flatten a ++ flatten b
flattenDFS (CVal x)     = [x]
flattenDFS CEra         = []
-- BLOCK 58:
flattenBFS :: Collapse a -> [a]
flattenBFS term = go term (SQ [] [] :: SQ (Collapse a)) where
  go (CSup k a b) sq = go CEra (sqPut b $ sqPut a $ sq)
  go (CVal x)     sq = x : go CEra sq
  go CEra         sq = case sqPop sq of
    Just (v,sq) -> go v sq
    Nothing     -> []
-- BLOCK 59:
flattenPQ :: Collapse a -> [a]
flattenPQ term = go term (PQLeaf :: PQ (Collapse a)) where
  go (CSup k a b) pq = go CEra (pqPut (k,a) $ pqPut (k,b) $ pq)
  go (CVal x)     pq = x : go CEra pq
  go CEra         pq = case pqPop pq of
    Just ((k,v),pq) -> go v pq
    Nothing         -> []
-- BLOCK 60:
flatten :: Collapse a -> [a]
flatten = flattenBFS
-- BLOCK 61:
-- Flat Collapser
-- --------------
-- BLOCK 62:
doCollapseFlatAt :: ReduceAt -> Book -> Loc -> HVM [Core]
doCollapseFlatAt reduceAt book host = do
  coll <- doCollapseAt reduceAt book host
  return $ flatten coll

-- BLOCK 63:
-- BLOCK 64:
module HVML.Compile where
-- BLOCK 65:
import Control.Monad (forM_, forM, foldM, when)
import Control.Monad.State
import Data.List
import Data.Word
import Debug.Trace
import HVML.Show
import HVML.Type hiding (fresh)
import qualified Data.Map.Strict as MS
-- BLOCK 66:
-- Compilation
-- -----------
-- BLOCK 67:
data CompileState = CompileState
  { next :: Word64
  , tabs :: Int
  , bins :: MS.Map String String  -- var_name => binder_host
  , vars :: [(String, String)]    -- [(var_name, var_host)]
  , code :: [String]
  }
-- BLOCK 68:
type Compile = State CompileState
-- BLOCK 69:
compile :: Book -> Word64 -> String
compile book fid =
  let full = compileWith compileFull book fid in
  let fast = compileWith compileFast book fid in
  let slow = compileWith compileSlow book fid in
  if "<ERR>" `isInfixOf` fast
    then unlines [ full , slow ]
    else unlines [ full , fast ]
-- BLOCK 70:
-- Compiles a function using either Fast-Mode or Full-Mode
compileWith :: (Book -> Word64 -> Core -> Bool -> [(Bool,String)] -> Compile ()) -> Book -> Word64 -> String
compileWith cmp book fid = 
  let copy   = fst (fst (mget (idToFunc book) fid)) in
  let args   = snd (fst (mget (idToFunc book) fid)) in
  let core   = snd (mget (idToFunc book) fid) in
  let state  = CompileState 0 0 MS.empty [] [] in
  let result = runState (cmp book fid core copy args) state in
  unlines $ reverse $ code (snd result)
-- BLOCK 71:
emit :: String -> Compile ()
emit line = modify $ \st -> st { code = (replicate (tabs st * 2) ' ' ++ line) : code st }
-- BLOCK 72:
tabInc :: Compile ()
tabInc = modify $ \st -> st { tabs = tabs st + 1 }
-- BLOCK 73:
tabDec :: Compile ()
tabDec = modify $ \st -> st { tabs = tabs st - 1 }
-- BLOCK 74:
bind :: String -> String -> Compile ()
bind var host = modify $ \st -> st { bins = MS.insert var host (bins st) }
-- BLOCK 75:
fresh :: String -> Compile String
fresh name = do
  uid <- gets next
  modify $ \s -> s { next = uid + 1 }
  return $ name ++ show uid
-- BLOCK 76:
-- Full Compiler
-- -------------
-- BLOCK 77:
compileFull :: Book -> Word64 -> Core -> Bool -> [(Bool,String)] -> Compile ()
compileFull book fid core copy args = do
  emit $ "Term " ++ mget (idToName book) fid ++ "_t(Term ref) {"
  tabInc
  forM_ (zip [0..] args) $ \(i, arg) -> do
    let argName = snd arg
    let argTerm = if fst arg
          then "reduce_at(term_loc(ref) + " ++ show i ++ ")"
          else "got(term_loc(ref) + " ++ show i ++ ")"
    bind argName argTerm
  result <- compileFullCore book fid core "root"
  st <- get
  forM_ (vars st) $ \ (var,host) -> do
    let varTerm = MS.findWithDefault "" var (bins st)
    emit $ "set(" ++ host ++ ", " ++ varTerm ++ ");"
  emit $ "return " ++ result ++ ";"
  tabDec
  emit "}"
-- BLOCK 78:
compileFullVar :: String -> String -> Compile String
compileFullVar var host = do
  bins <- gets bins
  case MS.lookup var bins of
    Just entry -> do
      return entry
    Nothing -> do
      modify $ \s -> s { vars = (var, host) : vars s }
      return "0"
-- BLOCK 79:
compileFullCore :: Book -> Word64 -> Core -> String -> Compile String
-- BLOCK 80:
compileFullCore book fid Era _ = do
  return $ "term_new(ERA, 0, 0)"
-- BLOCK 81:
compileFullCore book fid (Var name) host = do
  compileFullVar name host
-- BLOCK 82:
compileFullCore book fid (Let mode var val bod) host = do
  letNam <- fresh "let"
  emit $ "Loc " ++ letNam ++ " = alloc_node(2);"
  -- emit $ "set(" ++ letNam ++ " + 0, term_new(SUB, 0, 0));"
  valT <- compileFullCore book fid val (letNam ++ " + 0")
  emit $ "set(" ++ letNam ++ " + 0, " ++ valT ++ ");"
  bind var $ "term_new(VAR, 0, " ++ letNam ++ " + 0)"
  bodT <- compileFullCore book fid bod (letNam ++ " + 1")
  emit $ "set(" ++ letNam ++ " + 1, " ++ bodT ++ ");"
  return $ "term_new(LET, " ++ show (fromEnum mode) ++ ", " ++ letNam ++ ")"
-- BLOCK 83:
compileFullCore book fid (Lam var bod) host = do
  lamNam <- fresh "lam"
  emit $ "Loc " ++ lamNam ++ " = alloc_node(1);"
  -- emit $ "set(" ++ lamNam ++ " + 0, term_new(SUB, 0, 0));"
  bind var $ "term_new(VAR, 0, " ++ lamNam ++ " + 0)"
  bodT <- compileFullCore book fid bod (lamNam ++ " + 0")
  emit $ "set(" ++ lamNam ++ " + 0, " ++ bodT ++ ");"
  return $ "term_new(LAM, 0, " ++ lamNam ++ ")"
-- BLOCK 84:
compileFullCore book fid (App fun arg) host = do
  appNam <- fresh "app"
  emit $ "Loc " ++ appNam ++ " = alloc_node(2);"
  funT <- compileFullCore book fid fun (appNam ++ " + 0")
  argT <- compileFullCore book fid arg (appNam ++ " + 1")
  emit $ "set(" ++ appNam ++ " + 0, " ++ funT ++ ");"
  emit $ "set(" ++ appNam ++ " + 1, " ++ argT ++ ");"
  return $ "term_new(APP, 0, " ++ appNam ++ ")"
-- BLOCK 85:
compileFullCore book fid (Sup lab tm0 tm1) host = do
  supNam <- fresh "sup"
  emit $ "Loc " ++ supNam ++ " = alloc_node(2);"
  tm0T <- compileFullCore book fid tm0 (supNam ++ " + 0")
  tm1T <- compileFullCore book fid tm1 (supNam ++ " + 1")
  emit $ "set(" ++ supNam ++ " + 0, " ++ tm0T ++ ");"
  emit $ "set(" ++ supNam ++ " + 1, " ++ tm1T ++ ");"
  return $ "term_new(SUP, " ++ show lab ++ ", " ++ supNam ++ ")"
-- BLOCK 86:
compileFullCore book fid (Dup lab dp0 dp1 val bod) host = do
  dupNam <- fresh "dup"
  emit $ "Loc " ++ dupNam ++ " = alloc_node(2);"
  emit $ "set(" ++ dupNam ++ " + 1, term_new(SUB, 0, 0));"
  bind dp0 $ "term_new(DP0, " ++ show lab ++ ", " ++ dupNam ++ " + 0)"
  bind dp1 $ "term_new(DP1, " ++ show lab ++ ", " ++ dupNam ++ " + 0)"
  valT <- compileFullCore book fid val (dupNam ++ " + 0")
  emit $ "set(" ++ dupNam ++ " + 0, " ++ valT ++ ");"
  bodT <- compileFullCore book fid bod host
  return bodT
-- BLOCK 87:
compileFullCore book fid (Ctr cid fds) host = do
  ctrNam <- fresh "ctr"
  let arity = length fds
  emit $ "Loc " ++ ctrNam ++ " = alloc_node(" ++ show arity ++ ");"
  fdsT <- mapM (\ (i,fd) -> compileFullCore book fid fd (ctrNam ++ " + " ++ show i)) (zip [0..] fds)
  sequence_ [emit $ "set(" ++ ctrNam ++ " + " ++ show i ++ ", " ++ fdT ++ ");" | (i,fdT) <- zip [0..] fdsT]
  return $ "term_new(CTR, u12v2_new(" ++ show cid ++ ", " ++ show arity ++ "), " ++ ctrNam ++ ")"
-- BLOCK 88:
compileFullCore book fid tm@(Mat val mov css) host = do
  matNam <- fresh "mat"
  let arity = length css
  emit $ "Loc " ++ matNam ++ " = alloc_node(" ++ show (1 + arity) ++ ");"
  valT <- compileFullCore book fid val (matNam ++ " + 0")
  emit $ "set(" ++ matNam ++ " + 0, " ++ valT ++ ");"
  forM_ (zip [0..] css) $ \ (i,(ctr,fds,bod)) -> do
    -- Create a chain of lambdas for fields and moved vars
    let bod' = foldr Lam (foldr Lam bod (map fst mov)) fds
    bodT <- compileFullCore book fid bod' (matNam ++ " + " ++ show (i+1))
    emit $ "set(" ++ matNam ++ " + " ++ show (i+1) ++ ", " ++ bodT ++ ");"
  -- Create the base Mat term
  let mat = "term_new(MAT, u12v2_new(" ++ show arity ++ "," ++ show (ifLetLab book tm) ++ "), " ++ matNam ++ ")"
  -- Apply moved values
  foldM (\term (key, val) -> do
    appNam <- fresh "app"
    emit $ "Loc " ++ appNam ++ " = alloc_node(2);"
    valT <- compileFullCore book fid val (appNam ++ " + 1")
    emit $ "set(" ++ appNam ++ " + 0, " ++ term ++ ");"
    emit $ "set(" ++ appNam ++ " + 1, " ++ valT ++ ");"
    return $ "term_new(APP, 0, " ++ appNam ++ ")"
    ) mat mov
-- BLOCK 89:
compileFullCore book fid (U32 val) _ =
  return $ "term_new(W32, 0, " ++ show (fromIntegral val) ++ ")"
-- BLOCK 90:
compileFullCore book fid (Chr val) _ =
  return $ "term_new(CHR, 0, " ++ show (fromEnum val) ++ ")"
-- BLOCK 91:
compileFullCore book fid (Op2 opr nu0 nu1) host = do
  opxNam <- fresh "opx"
  emit $ "Loc " ++ opxNam ++ " = alloc_node(2);"
  nu0T <- compileFullCore book fid nu0 (opxNam ++ " + 0")
  nu1T <- compileFullCore book fid nu1 (opxNam ++ " + 1")
  emit $ "set(" ++ opxNam ++ " + 0, " ++ nu0T ++ ");"
  emit $ "set(" ++ opxNam ++ " + 1, " ++ nu1T ++ ");"
  return $ "term_new(OPX, " ++ show (fromEnum opr) ++ ", " ++ opxNam ++ ")"
-- BLOCK 92:
compileFullCore book fid (Ref rNam rFid rArg) host = do
  refNam <- fresh "ref"
  let arity = length rArg
  emit $ "Loc " ++ refNam ++ " = alloc_node(" ++ show arity ++ ");"
  argsT <- mapM (\ (i,arg) -> compileFullCore book fid arg (refNam ++ " + " ++ show i)) (zip [0..] rArg)
  sequence_ [emit $ "set(" ++ refNam ++ " + " ++ show i ++ ", " ++ argT ++ ");" | (i,argT) <- zip [0..] argsT]
  return $ "term_new(REF, u12v2_new(" ++ show rFid ++ ", " ++ show arity ++ "), " ++ refNam ++ ")"
-- BLOCK 93:
-- Fast Compiler
-- -------------
-- BLOCK 94:
-- Compiles a function using Fast-Mode
compileFast :: Book -> Word64 -> Core -> Bool -> [(Bool,String)] -> Compile ()
compileFast book fid core copy args = do
  emit $ "Term " ++ mget (idToName book) fid ++ "_f(Term ref) {"
  tabInc
  emit "u64 itrs = 0;"
  args <- forM (zip [0..] args) $ \ (i, (strict, arg)) -> do
    argNam <- fresh "arg"
    if strict then do
      emit $ "Term " ++ argNam ++ " = reduce_at(term_loc(ref) + " ++ show i ++ ");"
    else do
      emit $ "Term " ++ argNam ++ " = got(term_loc(ref) + " ++ show i ++ ");"
    if copy && strict then do
      case MS.lookup fid (idToLabs book) of
        Just labs -> do
          emit $ "if (term_tag(" ++ argNam ++ ") == ERA) {"
          emit $ "  return term_new(ERA, 0, 0);"
          emit $ "}"
          emit $ "if (term_tag(" ++ argNam ++ ") == SUP) {"
          tabInc
          emit $ "u64 lab = term_lab(" ++ argNam ++ ");"
          emit $ "if (1"
          forM_ (MS.keys labs) $ \lab -> do
            emit $ "    && lab != " ++ show lab
          emit $ ") {"
          tabInc
          emit $ "Term term = reduce_ref_sup(ref, " ++ show i ++ ");"
          emit $ "return term;"
          tabDec
          emit $ "}"
          tabDec
          emit $ "}"
        Nothing -> return ()
    else
      return ()
    bind arg argNam
    return argNam
  compileFastArgs book fid core args MS.empty
  tabDec
  emit "}"
-- BLOCK 95:
-- Compiles a fast function's argument list
compileFastArgs :: Book -> Word64 -> Core -> [String] -> MS.Map Int [String] -> Compile ()
compileFastArgs book fid body ctx reuse = do
  emit $ "while (1) {"
  tabInc
  compileFastBody book fid body ctx False 0 reuse
  tabDec
  emit $ "}"
-- BLOCK 96:
-- Compiles a fast function body (pattern-matching)
compileFastBody :: Book -> Word64 -> Core -> [String] -> Bool -> Int -> MS.Map Int [String] -> Compile ()
compileFastBody book fid term@(Mat val mov css) ctx stop@False itr reuse = do
  valT   <- compileFastCore book fid val reuse
  valNam <- fresh "val"
  numNam <- fresh "num"
  emit $ "Term " ++ valNam ++ " = (" ++ valT ++ ");"
  let isNumeric = length css > 0 && (let (ctr,fds,bod) = css !! 0 in ctr == "0")
-- BLOCK 97:
  -- Numeric Pattern-Matching
  if isNumeric then do
    emit $ "if (term_tag("++valNam++") == W32) {"
    tabInc
    emit $ "u32 " ++ numNam ++ " = term_loc(" ++ valNam ++ ");"
    emit $ "switch (" ++ numNam ++ ") {"
    tabInc
    forM_ (zip [0..] css) $ \ (i, (ctr,fds,bod)) -> do
      if i < length css - 1 then do
        emit $ "case " ++ show i ++ ": {"
        tabInc
        forM_ mov $ \ (key,val) -> do
          valT <- compileFastCore book fid val reuse
          bind key valT
        compileFastBody book fid bod ctx stop (itr + 1 + length mov) reuse
        emit $ "break;"
        tabDec
        emit $ "}"
      else do
        emit $ "default: {"
        tabInc
        preNam <- fresh "pre"
        emit $ "Term " ++ preNam ++ " = " ++ "term_new(W32, 0, "++numNam++" - "++show (length css - 1)++");"
        forM_ fds $ \ fd -> do
          bind fd preNam
        forM_ mov $ \ (key,val) -> do
          valT <- compileFastCore book fid val reuse
          bind key valT
        compileFastBody book fid bod ctx stop (itr + 1 + length fds + length mov) reuse
        emit $ "break;"
        tabDec
        emit $ "}"
    tabDec
    emit $ "}"
    tabDec
    emit $ "}"
-- BLOCK 98:
  -- Constructor Pattern-Matching (with IfLet)
  else do
    if ifLetLab book term > 0 then do
      emit $ "if (term_tag(" ++ valNam ++ ") == CTR) {"
      tabInc
      emit $ "if (u12v2_x(term_lab(" ++ valNam ++ ")) == " ++ show (ifLetLab book term - 1) ++ ") {"
      tabInc
      let (ctr,fds,bod) = css !! 0
      let reuse' = MS.insertWith (++) (length fds) ["term_loc(" ++ valNam ++ ")"] reuse
      forM_ (zip [0..] fds) $ \ (k,fd) -> do
        fdNam <- fresh "fd"
        emit $ "Term " ++ fdNam ++ " = got(term_loc(" ++ valNam ++ ") + " ++ show k ++ ");"
        bind fd fdNam
      forM_ mov $ \ (key,val) -> do
        valT <- compileFastCore book fid val reuse'
        bind key valT
      compileFastBody book fid bod ctx stop (itr + 1 + length fds + length mov) reuse'
      tabDec
      emit $ "} else {"
      tabInc
      let (ctr,fds,bod) = css !! 1
      when (length fds /= 1) $ do
        error "incorrect arity on if-let default case"
      fdNam <- fresh "fd"
      emit $ "Term " ++ fdNam ++ " = " ++ valNam ++ ";"
      bind (head fds) fdNam
      forM_ mov $ \ (key,val) -> do
        valT <- compileFastCore book fid val reuse
        bind key valT
      compileFastBody book fid bod ctx stop (itr + 1 + 1 + length mov) reuse
      tabDec
      emit $ "}"
      tabDec
      emit $ "}"
-- BLOCK 99:
    -- Constructor Pattern-Matching (without IfLet)
    else do
      emit $ "if (term_tag(" ++ valNam ++ ") == CTR) {"
      tabInc
      emit $ "switch (u12v2_x(term_lab(" ++ valNam ++ "))) {"
      tabInc
      forM_ (zip [0..] css) $ \ (i, (ctr,fds,bod)) -> do
        emit $ "case " ++ show i ++ ": {"
        tabInc
        let reuse' = MS.insertWith (++) (length fds) ["term_loc(" ++ valNam ++ ")"] reuse
        forM_ (zip [0..] fds) $ \ (k,fd) -> do
          fdNam <- fresh "fd"
          emit $ "Term " ++ fdNam ++ " = got(term_loc(" ++ valNam ++ ") + " ++ show k ++ ");"
          bind fd fdNam
        forM_ mov $ \ (key,val) -> do
          valT <- compileFastCore book fid val reuse'
          bind key valT
        compileFastBody book fid bod ctx stop (itr + 1 + length fds + length mov) reuse'
        emit $ "break;"
        tabDec
        emit $ "}"
      tabDec
      emit $ "}"
      tabDec
      emit $ "}"
-- BLOCK 100:
  compileFastUndo book fid term ctx itr reuse
-- BLOCK 101:
compileFastBody book fid term@(Dup lab dp0 dp1 val bod) ctx stop itr reuse = do
  valT <- compileFastCore book fid val reuse
  valNam <- fresh "val"
  dp0Nam <- fresh "dp0"
  dp1Nam <- fresh "dp1"
  emit $ "Term " ++ valNam ++ " = (" ++ valT ++ ");"
  emit $ "Term " ++ dp0Nam ++ ";"
  emit $ "Term " ++ dp1Nam ++ ";"
  emit $ "if (term_tag(" ++ valNam ++ ") == W32) {"
  tabInc
  emit $ "itrs += 1;"
  emit $ dp0Nam ++ " = " ++ valNam ++ ";"
  emit $ dp1Nam ++ " = " ++ valNam ++ ";"
  tabDec
  emit $ "} else {"
  tabInc
  dupNam <- fresh "dup"
  dupLoc <- compileFastAlloc 2 reuse
  emit $ "Loc " ++ dupNam ++ " = " ++ dupLoc ++ ";"
  emit $ "set(" ++ dupNam ++ " + 0, " ++ valNam ++ ");"
  emit $ "set(" ++ dupNam ++ " + 1, term_new(SUB, 0, 0));"
  emit $ dp0Nam ++ " = term_new(DP0, " ++ show lab ++ ", " ++ dupNam ++ " + 0);"
  emit $ dp1Nam ++ " = term_new(DP1, " ++ show lab ++ ", " ++ dupNam ++ " + 0);"
  tabDec
  emit $ "}"
  bind dp0 dp0Nam
  bind dp1 dp1Nam
  compileFastBody book fid bod ctx stop itr reuse
-- BLOCK 102:
compileFastBody book fid term@(Let mode var val bod) ctx stop itr reuse = do
  valT <- compileFastCore book fid val reuse
  case mode of
    LAZY -> do
      bind var valT
    STRI -> do
      case val of
        Ref _ rFid _ -> do
          valNam <- fresh "val"
          emit $ "Term " ++ valNam ++ " = reduce(" ++ mget (idToName book) rFid ++ "_f(" ++ valT ++ "));"
          bind var valNam
        _ -> do
          valNam <- fresh "val" 
          emit $ "Term " ++ valNam ++ " = reduce(" ++ valT ++ ");"
          bind var valNam
    PARA -> do -- TODO: implement parallel evaluation
      valNam <- fresh "val"
      emit $ "Term " ++ valNam ++ " = reduce(" ++ valT ++ ");"
      bind var valNam
  compileFastBody book fid bod ctx stop itr reuse
-- BLOCK 103:
compileFastBody book fid term@(Ref fNam fFid fArg) ctx stop itr reuse | fFid == fid = do
  forM_ (zip fArg ctx) $ \ (arg, ctxVar) -> do
    argT <- compileFastCore book fid arg reuse
    emit $ "" ++ ctxVar ++ " = " ++ argT ++ ";"
  emit $ "itrs += " ++ show (itr + 1) ++ ";"
  emit $ "continue;"
-- BLOCK 104:
compileFastBody book fid term ctx stop itr reuse = do
  emit $ "itrs += " ++ show itr ++ ";"
  body <- compileFastCore book fid term reuse
  compileFastSave book fid term ctx itr reuse
  emit $ "return " ++ body ++ ";"
-- BLOCK 105:
-- Falls back from fast mode to full mode
compileFastUndo :: Book -> Word64 -> Core -> [String] -> Int -> MS.Map Int [String] -> Compile ()
compileFastUndo book fid term ctx itr reuse = do
  forM_ (zip [0..] ctx) $ \ (i, arg) -> do
    emit $ "set(term_loc(ref) + "++show i++", " ++ arg ++ ");"
  emit $ "return " ++ mget (idToName book) fid ++ "_t(ref);"
-- BLOCK 106:
-- Completes a fast mode call
compileFastSave :: Book -> Word64 -> Core -> [String] -> Int -> MS.Map Int [String] -> Compile ()
compileFastSave book fid term ctx itr reuse = do
  emit $ "*HVM.itrs += itrs;"
-- BLOCK 107:
-- Helper function to allocate nodes with reuse
compileFastAlloc :: Int -> MS.Map Int [String] -> Compile String
compileFastAlloc arity reuse = do
  return $ "alloc_node(" ++ show arity ++ ")"
  -- FIXME: temporarily disabled, caused bug in:
  -- data List {
    -- #Nil
    -- #Cons{head tail}
  -- }
  -- @cat(xs ys) = ~xs !ys {
    -- #Nil: ys
    -- #Cons{h t}: #Cons{h @cat(t ys)}
  -- }
  -- @main = @cat(#Cons{1 #Nil} #Nil)
  -- case MS.lookup arity reuse of
    -- Just (loc:locs) -> return loc
    -- _ -> return $ "alloc_node(" ++ show arity ++ ")"
-- BLOCK 108:
-- Compiles a core term in fast mode
compileFastCore :: Book -> Word64 -> Core -> MS.Map Int [String] -> Compile String
-- BLOCK 109:
compileFastCore book fid Era reuse = 
  return $ "term_new(ERA, 0, 0)"
-- BLOCK 110:
compileFastCore book fid (Let mode var val bod) reuse = do
  valT <- compileFastCore book fid val reuse
  case mode of
    LAZY -> do
      emit $ "itrs += 1;"
      bind var valT
    STRI -> do
      valNam <- fresh "val"
      emit $ "itrs += 1;"
      emit $ "Term " ++ valNam ++ " = reduce(" ++ valT ++ ");"
      bind var valNam
    PARA -> do -- TODO: implement parallel evaluation
      valNam <- fresh "val"
      emit $ "Term " ++ valNam ++ " = reduce(" ++ valT ++ ");"
      bind var valNam
  compileFastCore book fid bod reuse
-- BLOCK 111:
compileFastCore book fid (Var name) reuse = do
  compileFastVar name
-- BLOCK 112:
compileFastCore book fid (Lam var bod) reuse = do
  lamNam <- fresh "lam"
  lamLoc <- compileFastAlloc 1 reuse
  emit $ "Loc " ++ lamNam ++ " = " ++ lamLoc ++ ";"
  -- emit $ "set(" ++ lamNam ++ " + 0, term_new(SUB, 0, 0));"
  bind var $ "term_new(VAR, 0, " ++ lamNam ++ " + 0)"
  bodT <- compileFastCore book fid bod reuse
  emit $ "set(" ++ lamNam ++ " + 0, " ++ bodT ++ ");"
  return $ "term_new(LAM, 0, " ++ lamNam ++ ")"
-- BLOCK 113:
compileFastCore book fid (App fun arg) reuse = do
  appNam <- fresh "app"
  appLoc <- compileFastAlloc 2 reuse
  emit $ "Loc " ++ appNam ++ " = " ++ appLoc ++ ";"
  funT <- compileFastCore book fid fun reuse
  argT <- compileFastCore book fid arg reuse
  emit $ "set(" ++ appNam ++ " + 0, " ++ funT ++ ");"
  emit $ "set(" ++ appNam ++ " + 1, " ++ argT ++ ");"
  return $ "term_new(APP, 0, " ++ appNam ++ ")"
-- BLOCK 114:
compileFastCore book fid (Sup lab tm0 tm1) reuse = do
  supNam <- fresh "sup"
  supLoc <- compileFastAlloc 2 reuse
  emit $ "Loc " ++ supNam ++ " = " ++ supLoc ++ ";"
  tm0T <- compileFastCore book fid tm0 reuse
  tm1T <- compileFastCore book fid tm1 reuse
  emit $ "set(" ++ supNam ++ " + 0, " ++ tm0T ++ ");"
  emit $ "set(" ++ supNam ++ " + 1, " ++ tm1T ++ ");"
  return $ "term_new(SUP, " ++ show lab ++ ", " ++ supNam ++ ")"
-- BLOCK 115:
compileFastCore book fid (Dup lab dp0 dp1 val bod) reuse = do
  dupNam <- fresh "dup"
  dp0Nam <- fresh "dp0"
  dp1Nam <- fresh "dp1"
  valNam <- fresh "val"
  valT   <- compileFastCore book fid val reuse
  emit $ "Term " ++ valNam ++ " = (" ++ valT ++ ");"
  emit $ "Term " ++ dp0Nam ++ ";"
  emit $ "Term " ++ dp1Nam ++ ";"
  emit $ "if (term_tag("++valNam++") == W32 || term_tag("++valNam++") == CHR) {"
  tabInc
  emit $ "itrs += 1;"
  emit $ dp0Nam ++ " = " ++ valNam ++ ";"
  emit $ dp1Nam ++ " = " ++ valNam ++ ";"
  tabDec
  emit $ "} else {"
  tabInc
  dupLoc <- compileFastAlloc 2 reuse
  emit $ "Loc " ++ dupNam ++ " = " ++ dupLoc ++ ";"
  emit $ "set(" ++ dupNam ++ " + 0, " ++ valNam ++ ");"
  emit $ "set(" ++ dupNam ++ " + 1, term_new(SUB, 0, 0));"
  emit $ dp0Nam ++ " = term_new(DP0, " ++ show lab ++ ", " ++ dupNam ++ " + 0);"
  emit $ dp1Nam ++ " = term_new(DP1, " ++ show lab ++ ", " ++ dupNam ++ " + 0);"
  tabDec
  emit $ "}"
  bind dp0 dp0Nam
  bind dp1 dp1Nam
  compileFastCore book fid bod reuse
-- BLOCK 116:
compileFastCore book fid (Ctr cid fds) reuse = do
  ctrNam <- fresh "ctr"
  let arity = length fds
  ctrLoc <- compileFastAlloc arity reuse
  emit $ "Loc " ++ ctrNam ++ " = " ++ ctrLoc ++ ";"
  fdsT <- mapM (\ (i,fd) -> compileFastCore book fid fd reuse) (zip [0..] fds)
  sequence_ [emit $ "set(" ++ ctrNam ++ " + " ++ show i ++ ", " ++ fdT ++ ");" | (i,fdT) <- zip [0..] fdsT]
  return $ "term_new(CTR, u12v2_new(" ++ show cid ++ ", " ++ show arity ++ "), " ++ ctrNam ++ ")"
-- BLOCK 117:
compileFastCore book fid tm@(Mat val mov css) reuse = do
  matNam <- fresh "mat"
  let arity = length css
  matLoc <- compileFastAlloc (1 + arity) reuse
  emit $ "Loc " ++ matNam ++ " = " ++ matLoc ++ ";"
  valT <- compileFastCore book fid val reuse
  emit $ "set(" ++ matNam ++ " + 0, " ++ valT ++ ");"
  forM_ (zip [0..] css) $ \ (i,(ctr,fds,bod)) -> do
    let bod' = foldr Lam (foldr Lam bod (map fst mov)) fds
    bodT <- compileFastCore book fid bod' reuse
    emit $ "set(" ++ matNam ++ " + " ++ show (i+1) ++ ", " ++ bodT ++ ");"
  let mat = "term_new(MAT, u12v2_new(" ++ show arity ++ "," ++ show (ifLetLab book tm) ++ "), " ++ matNam ++ ")"
  foldM (\term (key, val) -> do
    appNam <- fresh "app"
    appLoc <- compileFastAlloc 2 reuse
    emit $ "Loc " ++ appNam ++ " = " ++ appLoc ++ ";"
    valT <- compileFastCore book fid val reuse
    emit $ "set(" ++ appNam ++ " + 0, " ++ term ++ ");"
    emit $ "set(" ++ appNam ++ " + 1, " ++ valT ++ ");"
    return $ "term_new(APP, 0, " ++ appNam ++ ")"
    ) mat mov
-- BLOCK 118:
compileFastCore book fid (U32 val) reuse =
  return $ "term_new(W32, 0, " ++ show (fromIntegral val) ++ ")"
-- BLOCK 119:
compileFastCore book fid (Chr val) reuse =
  return $ "term_new(CHR, 0, " ++ show (fromEnum val) ++ ")"
-- BLOCK 120:
compileFastCore book fid (Op2 opr nu0 nu1) reuse = do
  opxNam <- fresh "opx"
  retNam <- fresh "ret"
  nu0Nam <- fresh "nu0"
  nu1Nam <- fresh "nu1"
  nu0T <- compileFastCore book fid nu0 reuse
  nu1T <- compileFastCore book fid nu1 reuse
  emit $ "Term " ++ nu0Nam ++ " = (" ++ nu0T ++ ");"
  emit $ "Term " ++ nu1Nam ++ " = (" ++ nu1T ++ ");"
  emit $ "Term " ++ retNam ++ ";"
  emit $ "if (term_tag(" ++ nu0Nam ++ ") == W32 && term_tag(" ++ nu1Nam ++ ") == W32) {"
  emit $ "  itrs += 2;"
  let oprStr = case opr of
        OP_ADD -> "+"
        OP_SUB -> "-"
        OP_MUL -> "*"
        OP_DIV -> "/"
        OP_MOD -> "%"
        OP_EQ  -> "=="
        OP_NE  -> "!="
        OP_LT  -> "<"
        OP_GT  -> ">"
        OP_LTE -> "<="
        OP_GTE -> ">="
        OP_AND -> "&"
        OP_OR  -> "|"
        OP_XOR -> "^"
        OP_LSH -> "<<"
        OP_RSH -> ">>"
  emit $ "  " ++ retNam ++ " = term_new(W32, 0, term_loc(" ++ nu0Nam ++ ") " ++ oprStr ++ " term_loc(" ++ nu1Nam ++ "));"
  emit $ "} else {"
  opxLoc <- compileFastAlloc 2 reuse
  emit $ "  Loc " ++ opxNam ++ " = " ++ opxLoc ++ ";"
  emit $ "  set(" ++ opxNam ++ " + 0, " ++ nu0Nam ++ ");"
  emit $ "  set(" ++ opxNam ++ " + 1, " ++ nu1Nam ++ ");"
  emit $ "  " ++ retNam ++ " = term_new(OPX, " ++ show (fromEnum opr) ++ ", " ++ opxNam ++ ");"
  emit $ "}"
  return $ retNam
-- BLOCK 121:
compileFastCore book fid (Ref rNam rFid rArg) reuse = do
-- BLOCK 122:
  -- Inline Dynamic SUP
  if rNam == "SUP" then do
    let [lab, tm0, tm1] = rArg
    supNam <- fresh "sup"
    labNam <- fresh "lab"
    supLoc <- compileFastAlloc 2 reuse
    labT <- compileFastCore book fid lab reuse
    emit $ "Term " ++ labNam ++ " = reduce(" ++ labT ++ ");"
    emit $ "if (term_tag(" ++ labNam ++ ") != W32) {"
    emit $ "  printf(\"ERROR:non-numeric-sup-label\\n\");"
    emit $ "}"
    emit $ "itrs += 1;"
    emit $ "Loc " ++ supNam ++ " = " ++ supLoc ++ ";"
    tm0T <- compileFastCore book fid tm0 reuse
    tm1T <- compileFastCore book fid tm1 reuse
    emit $ "set(" ++ supNam ++ " + 0, " ++ tm0T ++ ");"
    emit $ "set(" ++ supNam ++ " + 1, " ++ tm1T ++ ");"
    return $ "term_new(SUP, term_loc(" ++ labNam ++ "), " ++ supNam ++ ")"
-- BLOCK 123:
  -- Inline Dynamic DUP
  else if rNam == "DUP" && (case rArg of [_, _, Lam _ (Lam _ _)] -> True ; _ -> False) then do
    let [lab, val, Lam x (Lam y body)] = rArg
    dupNam <- fresh "dup"
    labNam <- fresh "lab"
    dupLoc <- compileFastAlloc 2 reuse
    labT <- compileFastCore book fid lab reuse
    emit $ "Term " ++ labNam ++ " = reduce(" ++ labT ++ ");"
    emit $ "if (term_tag(" ++ labNam ++ ") != W32) {"
    emit $ "  printf(\"ERROR:non-numeric-sup-label\\n\");"
    emit $ "}"
    emit $ "itrs += 3;"
    emit $ "Loc " ++ dupNam ++ " = " ++ dupLoc ++ ";"
    valT <- compileFastCore book fid val reuse
    emit $ "set(" ++ dupNam ++ " + 0, " ++ valT ++ ");"
    emit $ "set(" ++ dupNam ++ " + 1, term_new(SUB, 0, 0));"
    bind x $ "term_new(DP0, term_loc(" ++ labNam ++ "), " ++ dupNam ++ " + 0)"
    bind y $ "term_new(DP1, term_loc(" ++ labNam ++ "), " ++ dupNam ++ " + 0)"
    compileFastCore book fid body reuse
-- BLOCK 124:
  -- Create REF node
  else do
    refNam <- fresh "ref"
    let arity = length rArg
    refLoc <- compileFastAlloc arity reuse
    emit $ "Loc " ++ refNam ++ " = " ++ refLoc ++ ";"
    argsT <- mapM (\ (i,arg) -> compileFastCore book fid arg reuse) (zip [0..] rArg)
    sequence_ [emit $ "set(" ++ refNam ++ " + " ++ show i ++ ", " ++ argT ++ ");" | (i,argT) <- zip [0..] argsT]
    return $ "term_new(REF, u12v2_new(" ++ show rFid ++ ", " ++ show arity ++ "), " ++ refNam ++ ")"
-- BLOCK 125:
-- Compiles a variable in fast mode
compileFastVar :: String -> Compile String
compileFastVar var = do
  bins <- gets bins
  case MS.lookup var bins of
    Just entry -> do
      return entry
    Nothing -> do
      return "<ERR>"
-- BLOCK 126:
-- Compiles a function using Fast-Mode
compileSlow :: Book -> Word64 -> Core -> Bool -> [(Bool,String)] -> Compile ()
compileSlow book fid core copy args = do
  emit $ "Term " ++ mget (idToName book) fid ++ "_f(Term ref) {"
  emit $ "  return " ++ mget (idToName book) fid ++ "_t(ref);"
  emit $ "}"

-- BLOCK 127:
-- BLOCK 128:
module HVML.Extract where
-- BLOCK 129:
import Control.Monad (foldM)
import Control.Monad.State
import Data.Char (chr, ord)
import Data.IORef
import Data.Word
import HVML.Show
import HVML.Type
import System.IO.Unsafe (unsafeInterleaveIO)
import qualified Data.IntSet as IS
import qualified Data.Map.Strict as MS
import Debug.Trace
-- BLOCK 130:
extractCoreAt :: IORef IS.IntSet -> ReduceAt -> Book -> Loc -> HVM Core
-- BLOCK 131:
extractCoreAt dupsRef reduceAt book host = unsafeInterleaveIO $ do
  term <- reduceAt book host
  case tagT (termTag term) of
-- BLOCK 132:
    ERA -> do
      return Era
-- BLOCK 133:
    LET -> do
      let loc  = termLoc term
      let mode = modeT (termLab term)
      name <- return $ "$" ++ show (loc + 0)
      val  <- extractCoreAt dupsRef reduceAt book (loc + 0)
      bod  <- extractCoreAt dupsRef reduceAt book (loc + 1)
      return $ Let mode name val bod
-- BLOCK 134:
    LAM -> do
      let loc = termLoc term
      name <- return $ "$" ++ show (loc + 0)
      bod  <- extractCoreAt dupsRef reduceAt book (loc + 0)
      return $ Lam name bod
-- BLOCK 135:
    APP -> do
      let loc = termLoc term
      fun <- extractCoreAt dupsRef reduceAt book (loc + 0)
      arg <- extractCoreAt dupsRef reduceAt book (loc + 1)
      return $ App fun arg
-- BLOCK 136:
    SUP -> do
      let loc = termLoc term
      let lab = termLab term
      tm0 <- extractCoreAt dupsRef reduceAt book (loc + 0)
      tm1 <- extractCoreAt dupsRef reduceAt book (loc + 1)
      return $ Sup lab tm0 tm1
-- BLOCK 137:
    VAR -> do
      let loc = termLoc term
      sub <- got (loc + 0)
      if termGetBit sub == 0
        then do
          name <- return $ "$" ++ show (loc + 0)
          return $ Var name
        else do
          set (loc + 0) (termRemBit sub)
          extractCoreAt dupsRef reduceAt book (loc + 0)
-- BLOCK 138:
    DP0 -> do
      let loc = termLoc term
      let lab = termLab term
      dups <- readIORef dupsRef
      if IS.member (fromIntegral loc) dups
      then do
        name <- return $ "$" ++ show (loc + 0)
        return $ Var name
      else do
        dp0 <- return $ "$" ++ show (loc + 0)
        dp1 <- return $ "$" ++ show (loc + 1)
        val <- extractCoreAt dupsRef reduceAt book loc
        modifyIORef' dupsRef (IS.insert (fromIntegral loc))
        return $ Dup lab dp0 dp1 val (Var dp0)
-- BLOCK 139:
    DP1 -> do
      let loc = termLoc term
      let lab = termLab term
      dups <- readIORef dupsRef
      if IS.member (fromIntegral loc) dups
      then do
        name <- return $ "$" ++ show (loc + 1)
        return $ Var name
      else do
        dp0 <- return $ "$" ++ show (loc + 0)
        dp1 <- return $ "$" ++ show (loc + 1)
        val <- extractCoreAt dupsRef reduceAt book loc
        modifyIORef' dupsRef (IS.insert (fromIntegral loc))
        return $ Dup lab dp0 dp1 val (Var dp1)
-- BLOCK 140:
    CTR -> do
      let loc = termLoc term
      let lab = termLab term
      let cid = u12v2X lab
      let ari = u12v2Y lab
      let ars = if ari == 0 then [] else [0..ari-1]
      fds <- mapM (\i -> extractCoreAt dupsRef reduceAt book (loc + i)) ars
      return $ Ctr cid fds
-- BLOCK 141:
    MAT -> do
      let loc = termLoc term
      let len = u12v2X $ termLab term
      val <- extractCoreAt dupsRef reduceAt book (loc + 0)
      css <- mapM (\i -> extractCoreAt dupsRef reduceAt book (loc + 1 + i)) [0..len-1]
      css <- mapM (\c -> return ("#", [], c)) css -- FIXME: recover names and fields on extraction (must store id)
      return $ Mat val [] css
-- BLOCK 142:
    W32 -> do
      let val = termLoc term
      return $ U32 (fromIntegral val)
-- BLOCK 143:
    CHR -> do
      let val = termLoc term
      return $ Chr (chr (fromIntegral val))
-- BLOCK 144:
    OPX -> do
      let loc = termLoc term
      let opr = toEnum (fromIntegral (termLab term))
      nm0 <- extractCoreAt dupsRef reduceAt book (loc + 0)
      nm1 <- extractCoreAt dupsRef reduceAt book (loc + 1)
      return $ Op2 opr nm0 nm1
-- BLOCK 145:
    OPY -> do
      let loc = termLoc term
      let opr = toEnum (fromIntegral (termLab term))
      nm0 <- extractCoreAt dupsRef reduceAt book (loc + 0)
      nm1 <- extractCoreAt dupsRef reduceAt book (loc + 1)
      return $ Op2 opr nm0 nm1
-- BLOCK 146:
    REF -> do
      let loc = termLoc term
      let lab = termLab term
      let fid = u12v2X lab
      let ari = u12v2Y lab
      let aux = if ari == 0 then [] else [0..ari-1]
      arg <- mapM (\i -> extractCoreAt dupsRef reduceAt book (loc + i)) aux
      let name = MS.findWithDefault "?" fid (idToName book)
      return $ Ref name fid arg
-- BLOCK 147:
    _ -> do
      return Era
-- BLOCK 148:
doExtractCoreAt :: ReduceAt -> Book -> Loc -> HVM Core
doExtractCoreAt reduceAt book loc = do
  dupsRef <- newIORef IS.empty
  core    <- extractCoreAt dupsRef reduceAt book loc
  return core
  -- return $ doLiftDups core
-- BLOCK 149:
-- Lifting Dups
-- ------------
-- BLOCK 150:
liftDups :: Core -> (Core, Core -> Core)
-- BLOCK 151:
liftDups (Var nam) =
  (Var nam, id)
-- BLOCK 152:
liftDups (Ref nam fid arg) =
  let (argT, argD) = liftDupsList arg
  in (Ref nam fid argT, argD)
-- BLOCK 153:
liftDups Era =
  (Era, id)
-- BLOCK 154:
liftDups (Lam str bod) =
  let (bodT, bodD) = liftDups bod
  in (Lam str bodT, bodD)
-- BLOCK 155:
liftDups (App fun arg) =
  let (funT, funD) = liftDups fun
      (argT, argD) = liftDups arg
  in (App funT argT, funD . argD)
-- BLOCK 156:
liftDups (Sup lab tm0 tm1) =
  let (tm0T, tm0D) = liftDups tm0
      (tm1T, tm1D) = liftDups tm1
  in (Sup lab tm0T tm1T, tm0D . tm1D)
-- BLOCK 157:
liftDups (Dup lab dp0 dp1 val bod) =
  let (valT, valD) = liftDups val
      (bodT, bodD) = liftDups bod
  in (bodT, \x -> valD (bodD (Dup lab dp0 dp1 valT x)))
-- BLOCK 158:
liftDups (Ctr cid fds) =
  let (fdsT, fdsD) = liftDupsList fds
  in (Ctr cid fdsT, fdsD)
-- BLOCK 159:
liftDups (Mat val mov css) =
  let (valT, valD) = liftDups val
      (movT, movD) = liftDupsMov mov
      (cssT, cssD) = liftDupsCss css
  in (Mat valT movT cssT, valD . movD . cssD)
-- BLOCK 160:
liftDups (U32 val) =
  (U32 val, id)
-- BLOCK 161:
liftDups (Chr val) =
  (Chr val, id)
-- BLOCK 162:
liftDups (Op2 opr nm0 nm1) =
  let (nm0T, nm0D) = liftDups nm0
      (nm1T, nm1D) = liftDups nm1
  in (Op2 opr nm0T nm1T, nm0D . nm1D)
-- BLOCK 163:
liftDups (Let mod nam val bod) =
  let (valT, valD) = liftDups val
      (bodT, bodD) = liftDups bod
  in (Let mod nam valT bodT, valD . bodD)
-- BLOCK 164:
liftDupsList :: [Core] -> ([Core], Core -> Core)
-- BLOCK 165:
liftDupsList [] = 
  ([], id)
-- BLOCK 166:
liftDupsList (x:xs) =
  let (xT, xD)   = liftDups x
      (xsT, xsD) = liftDupsList xs
  in (xT:xsT, xD . xsD)
-- BLOCK 167:
liftDupsMov :: [(String, Core)] -> ([(String, Core)], Core -> Core)
-- BLOCK 168:
liftDupsMov [] = 
  ([], id)
-- BLOCK 169:
liftDupsMov ((k,v):xs) =
  let (vT, vD)   = liftDups v
      (xsT, xsD) = liftDupsMov xs
  in ((k,vT):xsT, vD . xsD)
-- BLOCK 170:
liftDupsCss :: [(String, [String], Core)] -> ([(String, [String], Core)], Core -> Core)
-- BLOCK 171:
liftDupsCss [] = 
  ([], id)
-- BLOCK 172:
liftDupsCss ((c,fs,b):xs) =
  let (bT, bD)   = liftDups b
      (xsT, xsD) = liftDupsCss xs
  in ((c,fs,bT):xsT, bD . xsD)
-- BLOCK 173:
doLiftDups :: Core -> Core
doLiftDups term =
  let (termExpr, termDups) = liftDups term in
  let termBody = termDups (Var "") in
  -- hack to print expr before dups
  Let LAZY "" termExpr termBody

-- BLOCK 174:
-- BLOCK 175:
module HVML.Inject where
-- BLOCK 176:
import Control.Monad (foldM, when, forM_)
import Control.Monad.State
import Data.Char (ord)
import Data.Word
import HVML.Show
import HVML.Type
import qualified Data.Map.Strict as Map
-- BLOCK 177:
type InjectM a = StateT InjectState HVM a
-- BLOCK 178:
data InjectState = InjectState
  { args :: Map.Map String Term -- maps var names to binder locations
  , vars :: [(String, Loc)]     -- list of (var name, usage location) pairs
  }
-- BLOCK 179:
emptyState :: InjectState
emptyState = InjectState Map.empty []
-- BLOCK 180:
injectCore :: Book -> Core -> Loc -> InjectM ()
-- BLOCK 181:
injectCore _ Era loc = do
  lift $ set loc (termNew _ERA_ 0 0)
-- BLOCK 182:
injectCore _ (Var nam) loc = do
  argsMap <- gets args
  case Map.lookup nam argsMap of
    Just term -> do
      lift $ set loc term
      when (head nam /= '&') $ do
        modify $ \s -> s { args = Map.delete nam (args s) }
    Nothing -> do
      modify $ \s -> s { vars = (nam, loc) : vars s }
-- BLOCK 183:
injectCore book (Let mod nam val bod) loc = do
  let_node <- lift $ allocNode 2
  modify $ \s -> s { args = Map.insert nam (termNew _VAR_ 0 (let_node + 0)) (args s) }
  injectCore book val (let_node + 0)
  injectCore book bod (let_node + 1)
  lift $ set loc (termNew _LET_ (fromIntegral $ fromEnum mod) let_node)
-- BLOCK 184:
injectCore book (Lam vr0 bod) loc = do
  lam <- lift $ allocNode 1
  -- lift $ set (lam + 0) (termNew _SUB_ 0 0)
  modify $ \s -> s { args = Map.insert vr0 (termNew _VAR_ 0 (lam + 0)) (args s) }
  injectCore book bod (lam + 0)
  lift $ set loc (termNew _LAM_ 0 lam)
-- BLOCK 185:
injectCore book (App fun arg) loc = do
  app <- lift $ allocNode 2
  injectCore book fun (app + 0)
  injectCore book arg (app + 1)
  lift $ set loc (termNew _APP_ 0 app)
-- BLOCK 186:
injectCore book (Sup lab tm0 tm1) loc = do
  sup <- lift $ allocNode 2
  injectCore book tm0 (sup + 0)
  injectCore book tm1 (sup + 1)
  lift $ set loc (termNew _SUP_ lab sup)
-- BLOCK 187:
injectCore book (Dup lab dp0 dp1 val bod) loc = do
  dup <- lift $ allocNode 2
  -- lift $ set (dup + 0) (termNew _SUB_ 0 0)
  lift $ set (dup + 1) (termNew _SUB_ 0 0)
  modify $ \s -> s 
    { args = Map.insert dp0 (termNew _DP0_ lab dup) 
           $ Map.insert dp1 (termNew _DP1_ lab dup) (args s) 
    }
  injectCore book val (dup + 0)
  injectCore book bod loc
-- BLOCK 188:
injectCore book (Ref nam fid arg) loc = do
  -- lift $ set loc (termNew _REF_ 0 fid)
  let arity = length arg
  ref <- lift $ allocNode (fromIntegral arity)
  sequence_ [injectCore book x (ref + i) | (i,x) <- zip [0..] arg]
  lift $ set loc (termNew _REF_ (u12v2New fid (fromIntegral arity)) ref)
-- BLOCK 189:
injectCore book (Ctr cid fds) loc = do
  let arity = length fds
  ctr <- lift $ allocNode (fromIntegral arity)
  sequence_ [injectCore book fd (ctr + ix) | (ix,fd) <- zip [0..] fds]
  lift $ set loc (termNew _CTR_ (u12v2New cid (fromIntegral arity)) ctr)
-- BLOCK 190:
injectCore book tm@(Mat val mov css) loc = do
  -- Allocate space for the Mat term
  mat <- lift $ allocNode (1 + fromIntegral (length css))
  -- Inject the value being matched
  injectCore book val (mat + 0)
  -- Inject each case body
  forM_ (zip [0..] css) $ \ (idx, (ctr, fds, bod)) -> do
    injectCore book (foldr Lam (foldr Lam bod (map fst mov)) fds) (mat + 1 + idx)
  -- After processing all cases, create the Mat term
  trm <- return $ termNew _MAT_ (u12v2New (fromIntegral (length css)) (ifLetLab book tm)) mat
  ret <- foldM (\mat (_, val) -> do
      app <- lift $ allocNode 2
      lift $ set (app + 0) mat
      injectCore book val (app + 1)
      return $ termNew _APP_ 0 app)
    trm
    mov
  lift $ set loc ret
-- BLOCK 191:
injectCore book (U32 val) loc = do
  lift $ set loc (termNew _W32_ 0 (fromIntegral val))
-- BLOCK 192:
injectCore book (Chr val) loc = do
  lift $ set loc (termNew _CHR_ 0 (fromIntegral $ ord val))
-- BLOCK 193:
injectCore book (Op2 opr nm0 nm1) loc = do
  opx <- lift $ allocNode 2
  injectCore book nm0 (opx + 0)
  injectCore book nm1 (opx + 1)
  lift $ set loc (termNew _OPX_ (fromIntegral $ fromEnum opr) opx)
-- BLOCK 194:
doInjectCoreAt :: Book -> Core -> Loc -> [(String,Term)] -> HVM Term
doInjectCoreAt book core host argList = do
  (_, state) <- runStateT (injectCore book core host) (emptyState { args = Map.fromList argList })
  foldM (\m (name, loc) -> do
    case Map.lookup name (args state) of
      Just term -> do
        set loc term
        if (head name /= '&') then do
          return $ Map.delete name m
        else do
          return $ m
      Nothing -> do
        error $ "Unbound variable: " ++ name)
    (args state)
    (vars state)
  got host

-- BLOCK 195:
-- Type.hs:
-- BLOCK 196:
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}
-- BLOCK 197:
module Main where
-- BLOCK 198:
import Control.Monad (when, forM_)
import Data.FileEmbed
import Data.Time.Clock
import Data.Word
import Foreign.C.Types
import Foreign.LibFFI
import Foreign.LibFFI.Types
import GHC.Conc
import HVML.Collapse
import HVML.Compile
import HVML.Extract
import HVML.Inject
import HVML.Parse
import HVML.Reduce
import HVML.Show
import HVML.Type
import System.CPUTime
import System.Environment (getArgs)
import System.Exit (exitWith, ExitCode(ExitSuccess, ExitFailure))
import System.IO
import System.IO (readFile)
import System.Posix.DynamicLinker
import System.Process (callCommand)
import Text.Printf
import qualified Data.Map.Strict as MS
-- BLOCK 199:
runtime_c :: String
runtime_c = $(embedStringFile "./src/HVML/Runtime.c")
-- BLOCK 200:
-- Main
-- ----
-- BLOCK 201:
data RunMode
  = Normalize
  | Collapse
  | Search
  deriving Eq
-- BLOCK 202:
main :: IO ()
main = do
  args <- getArgs
  result <- case args of
    ("run" : file : args) -> do
      let compiled = "-c" `elem` args
      let collapse = "-C" `elem` args
      let search   = "-S" `elem` args
      let stats    = "-s" `elem` args
      let debug    = "-d" `elem` args
      let mode | collapse  = Collapse
               | search    = Search
               | otherwise = Normalize
      cliRun file debug compiled mode stats
    ["help"] -> printHelp
    _ -> printHelp
  case result of
    Left err -> do
      putStrLn err
      exitWith (ExitFailure 1)
    Right _ -> do
      exitWith ExitSuccess
-- BLOCK 203:
printHelp :: IO (Either String ())
printHelp = do
  putStrLn "HVM-Lazy usage:"
  putStrLn "  hvml help       # Shows this help message"
  putStrLn "  hvml run <file> # Evals main"
  putStrLn "    -t # Returns the type (experimental)"
  putStrLn "    -c # Runs with compiled mode (fast)"
  putStrLn "    -C # Collapse the result to a list of λ-Terms"
  putStrLn "    -S # Search (collapse, then print the 1st λ-Term)"
  putStrLn "    -s # Show statistics"
  putStrLn "    -d # Print execution steps (debug mode)"
  return $ Right ()
-- BLOCK 204:
-- CLI Commands
-- ------------
-- BLOCK 205:
cliRun :: FilePath -> Bool -> Bool -> RunMode -> Bool -> IO (Either String ())
cliRun filePath debug compiled mode showStats = do
  -- Initialize the HVM
  hvmInit
  -- TASK: instead of parsing a core term out of the file, lets parse a Book.
  code <- readFile filePath
  book <- doParseBook code
  -- Create the C file content
  let funcs = map (\ (fid, _) -> compile book fid) (MS.toList (idToFunc book))
  let mainC = unlines $ [runtime_c] ++ funcs ++ [genMain book]
  -- Compile to native
  when compiled $ do
    -- Write the C file
    writeFile "./.main.c" mainC
    -- Compile to shared library
    callCommand "gcc -O2 -fPIC -shared .main.c -o .main.so"
    -- Load the dynamic library
    bookLib <- dlopen "./.main.so" [RTLD_NOW]
    -- Remove both generated files
    callCommand "rm .main.so"
    -- Register compiled functions
    forM_ (MS.keys (idToFunc book)) $ \ fid -> do
      funPtr <- dlsym bookLib (mget (idToName book) fid ++ "_f")
      hvmDefine fid funPtr
    -- Link compiled state
    hvmGotState <- hvmGetState
    hvmSetState <- dlsym bookLib "hvm_set_state"
    callFFI hvmSetState retVoid [argPtr hvmGotState]
  -- Abort when main isn't present
  when (not $ MS.member "main" (nameToId book)) $ do
    putStrLn "Error: 'main' not found."
    exitWith (ExitFailure 1)
  -- Normalize main
  init <- getCPUTime
  root <- doInjectCoreAt book (Ref "main" (mget (nameToId book) "main") []) 0 []
  rxAt <- if compiled
    then return (reduceCAt debug)
    else return (reduceAt debug)
  vals <- if mode == Collapse || mode == Search
    then doCollapseFlatAt rxAt book 0
    else do
      core <- doExtractCoreAt rxAt book 0
      return [(doLiftDups core)]
  -- Print all collapsed results
  when (mode == Collapse) $ do
    forM_ vals $ \ term -> do
      putStrLn $ showCore term
  -- Prints just the first collapsed result
  when (mode == Search || mode == Normalize) $ do
    putStrLn $ showCore (head vals)
  when (mode /= Normalize) $ do
    putStrLn ""
  -- Prints total time
  end <- getCPUTime
  -- Show stats
  when showStats $ do
    itrs <- getItr
    size <- getLen
    let time = fromIntegral (end - init) / (10^12) :: Double
    let mips = (fromIntegral itrs / 1000000.0) / time
    printf "WORK: %llu interactions\n" itrs
    printf "TIME: %.7f seconds\n" time
    printf "SIZE: %llu nodes\n" size
    printf "PERF: %.3f MIPS\n" mips
    return ()
  -- Finalize
  hvmFree
  return $ Right ()
-- BLOCK 206:
genMain :: Book -> String
genMain book =
  let mainFid = mget (nameToId book) "main"
      registerFuncs = unlines ["  hvm_define(" ++ show fid ++ ", " ++ mget (idToName book) fid ++ "_f);" | fid <- MS.keys (idToFunc book)]
  in unlines
    [ "int main() {"
    , "  hvm_init();"
    , registerFuncs
    , "  clock_t start = clock();"
    , "  Term root = term_new(REF, u12v2_new("++show mainFid++",0), 0);"
    , "  normal(root);"
    , "  double time = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;"
    , "  printf(\"WORK: %llu interactions\\n\", get_itr());"
    , "  printf(\"TIME: %.3fs seconds\\n\", time / 1000.0);"
    , "  printf(\"SIZE: %u nodes\\n\", get_len());"
    , "  printf(\"PERF: %.3f MIPS\\n\", (get_itr() / 1000000.0) / (time / 1000.0));"
    , "  hvm_free();"
    , "  return 0;"
    , "}"
    ]

-- BLOCK 207:
-- BLOCK 208:
module HVML.Parse where
-- BLOCK 209:
import Control.Monad (foldM, forM)
import Control.Monad.State
import Data.Either (isLeft)
import Data.List
import Data.Maybe
import Data.Word
import Debug.Trace
import HVML.Show
import HVML.Type
import Highlight (highlightError)
import System.Console.ANSI
import System.Exit (exitFailure)
import System.IO.Unsafe (unsafePerformIO)
import Text.Parsec hiding (State)
import Text.Parsec.Error
import Text.Parsec.Pos
import Text.Parsec.String
import qualified Data.Map.Strict as MS
-- BLOCK 210:
-- Core Parsers
-- ------------
-- BLOCK 211:
data ParserState = ParserState
  { parsedCtrToAri :: MS.Map String Int
  , parsedCtrToCid :: MS.Map String Word64
  , freshLabel     :: Word64
  }
-- BLOCK 212:
type ParserM = Parsec String ParserState
-- BLOCK 213:
parseCore :: ParserM Core
parseCore = do
  skip
  head <- lookAhead anyChar
  case head of
-- BLOCK 214:
    '*' -> do
      consume "*"
      return Era
-- BLOCK 215:
    'λ' -> do
      consume "λ"
      vr0 <- parseName1
      bod <- parseCore
      return $ Lam vr0 bod
-- BLOCK 216:
    '(' -> do
      next <- lookAhead (anyChar >> anyChar)
      case next of
        '+' -> parseOper OP_ADD
        '-' -> parseOper OP_SUB
        '*' -> parseOper OP_MUL
        '/' -> parseOper OP_DIV
        '%' -> parseOper OP_MOD
        '=' -> parseOper OP_EQ
        '!' -> parseOper OP_NE
        '&' -> parseOper OP_AND
        '|' -> parseOper OP_OR
        '^' -> parseOper OP_XOR
        '<' -> do
          next <- lookAhead (anyChar >> anyChar >> anyChar)
          case next of
            '<' -> parseOper OP_LSH
            '=' -> parseOper OP_LTE
            _   -> parseOper OP_LT
        '>' -> do
          next <- lookAhead (anyChar >> anyChar >> anyChar)
          case next of
            '>' -> parseOper OP_RSH
            '=' -> parseOper OP_GTE
            _   -> parseOper OP_GT
        _ -> do
          consume "("
          fun <- parseCore
          args <- many $ do
            closeWith ")"
            parseCore
          char ')'
          return $ foldl App fun args
-- BLOCK 217:
    '@' -> do
      parseRef
-- BLOCK 218:
    '&' -> do
      consume "&"
      name <- parseName
      next <- optionMaybe $ try $ char '{'
      case next of
        Just _ -> do
          tm0 <- parseCore
          tm1 <- parseCore
          consume "}"
          if null name then do
            num <- genFreshLabel
            return $ Sup num tm0 tm1
          else case reads name of
            [(num :: Word64, "")] -> do
              return $ Sup num tm0 tm1
            otherwise -> do
              return $ Ref "SUP" _SUP_F_ [Var ("&" ++ name), tm0, tm1]
        Nothing -> do
          return $ Var ("&" ++ name)
-- BLOCK 219:
    '!' -> do
      consume "!"
      skip
      next <- lookAhead anyChar
      case next of
-- BLOCK 220:
        '&' -> do
          consume "&"
          nam <- parseName
          consume "{"
          dp0 <- parseName1
          dp1 <- parseName1
          consume "}"
          consume "="
          val <- parseCore
          bod <- parseCore
          if null nam then do
            num <- genFreshLabel
            return $ Dup num dp0 dp1 val bod
          else case reads nam of
            [(num :: Word64, "")] -> do
              return $ Dup num dp0 dp1 val bod
            otherwise -> do
              return $ Ref "DUP" _DUP_F_ [Var ("&" ++ nam), val, Lam dp0 (Lam dp1 bod)]
-- BLOCK 221:
        '!' -> do
          consume "!"
          nam <- optionMaybe $ try $ do
            nam <- parseName1
            consume "="
            return nam
          val <- parseCore
          bod <- parseCore
          case nam of
            Just nam -> return $ Let STRI nam val bod
            Nothing  -> return $ Let STRI "_" val bod
-- BLOCK 222:
        '^' -> do
          consume "^"
          nam <- parseName1
          consume "="
          val <- parseCore
          bod <- parseCore
          return $ Let PARA nam val bod
-- BLOCK 223:
        _ -> do
          nam <- parseName1
          consume "="
          val <- parseCore
          bod <- parseCore
          return $ Let LAZY nam val bod
-- BLOCK 224:
    '#' -> parseCtr
-- BLOCK 225:
    '~' -> parseMat
-- BLOCK 226:
    '[' -> parseLst
-- BLOCK 227:
    '\'' -> parseChr
-- BLOCK 228:
    '"' -> parseStr
-- BLOCK 229:
    _ -> do
      name <- parseName1
      case reads (filter (/= '_') name) of
        [(num, "")] -> return $ U32 (fromIntegral (num :: Integer))
        _           -> return $ Var name
-- BLOCK 230:
parseRef :: ParserM Core
parseRef = do
  consume "@"
  name <- parseName1
  args <- option [] $ do
    try $ string "("
    args <- many $ do
      closeWith ")"
      parseCore
    consume ")"
    return args
  return $ Ref name 0 args
-- BLOCK 231:
parseCtr :: ParserM Core
parseCtr = do
  consume "#"
  nam <- parseName1
  cid <- if length nam == 0
    then return 0
    else do
      cids <- parsedCtrToCid <$> getState
      case MS.lookup nam cids of
        Just id -> return id
        Nothing -> case reads nam of
          [(num, "")] -> return (fromIntegral (num :: Integer))
          otherwise   -> fail $ "Unknown constructor: " ++ nam
  fds <- option [] $ do
    try $ consume "{"
    fds <- many $ do
      closeWith "}"
      parseCore
    consume "}"
    return fds
  return $ Ctr cid fds
-- BLOCK 232:
parseMat :: ParserM Core
parseMat = do
  consume "~"
  val <- parseCore
  -- Parse mov (external variables)
  mov <- many $ do
    try $ do
      skip
      consume "!"
    key <- parseName1
    val <- optionMaybe $ do
      try $ consume "="
      parseCore
    case val of
      Just v  -> return (key, v)
      Nothing -> return (key, Var key)
  consume "{"
  css <- many $ do
    closeWith "}"
    skip
    next <- lookAhead anyChar
    -- Parse constructor case
    if next == '#' then do
      consume "#"
      ctr <- parseName1
      fds <- option [] $ do
        try $ consume "{"
        fds <- many $ do
          closeWith "}"
          parseName1
        consume "}"
        return fds
      consume ":"
      bod <- parseCore
      return (ctr, fds, bod)
    -- Parse numeric or default case
    else do
      nam <- parseName1
      case reads nam of
        -- Numeric case
        [(n :: Word64, "")] -> do
          consume ":"
          bod <- parseCore
          return (nam, [], bod)
        -- Default case
        otherwise -> do
          consume ":"
          bod <- parseCore
          return ("_", [nam], bod)
  consume "}"
  css <- forM css $ \ (ctr, fds, bod) -> do
    cid <- case reads ctr of
      [(num, "")] -> do
        return $ Left (read num :: Word64)
      otherwise -> do
        st <- getState
        return $ Right $ fromMaybe maxBound $ MS.lookup ctr (parsedCtrToCid st)
    return (cid, (ctr, fds, bod))
  css <- return $ map snd $ sortOn fst css
  -- Transform matches with default cases into nested chain of matches
  if length css == 1 && (let (ctr, _, _) = head css in ctr == "_") then do
    fail "Match with only a default case is not allowed."
  else if (let (ctr, _, _) = last css in ctr == "_") then do
    let defName = (let (_,[nm],_) = last css in nm)
    let ifLets  = intoIfLetChain (Var defName) mov (init css) defName (last css)
    return $ Let LAZY defName val ifLets
  else do
    return $ Mat val mov css
-- BLOCK 233:
intoIfLetChain :: Core -> [(String, Core)] -> [(String, [String], Core)] -> String -> (String, [String], Core) -> Core
intoIfLetChain _ _ [] defName (_,_,defBody) = defBody
intoIfLetChain val mov ((ctr,fds,bod):css) defName defCase =
  let rest = intoIfLetChain val mov css defName defCase in 
  Mat val mov [(ctr, fds, bod), ("_", [defName], rest)]
-- BLOCK 234:
parseOper :: Oper -> ParserM Core
parseOper op = do
  consume "("
  consume (operToString op)
  nm0 <- parseCore
  nm1 <- parseCore
  consume ")"
  return $ Op2 op nm0 nm1
-- BLOCK 235:
parseEscapedChar :: ParserM Char
parseEscapedChar = choice
  [ try $ do
      char '\\'
      c <- oneOf "\\\"nrtbf0/\'"
      return $ case c of
        '\\' -> '\\'
        '/'  -> '/'
        '"'  -> '"'
        '\'' -> '\''
        'n'  -> '\n'
        'r'  -> '\r'
        't'  -> '\t'
        'b'  -> '\b'
        'f'  -> '\f'
        '0'  -> '\0'
  , try $ do
      string "\\u"
      code <- count 4 hexDigit
      return $ toEnum (read ("0x" ++ code) :: Int)
  , noneOf "\"\\"
  ]
-- BLOCK 236:
parseChr :: ParserM Core
parseChr = do
  skip
  char '\''
  c <- parseEscapedChar
  char '\''
  return $ Chr c
-- BLOCK 237:
parseStr :: ParserM Core
parseStr = do
  skip
  char '"'
  str <- many (noneOf "\"")
  char '"'
  return $ foldr (\c acc -> Ctr 1 [Chr c, acc]) (Ctr 0 []) str
-- BLOCK 238:
parseLst :: ParserM Core
parseLst = do
  skip
  char '['
  elems <- many $ do
    closeWith "]"
    parseCore
  char ']'
  return $ foldr (\x acc -> Ctr 1 [x, acc]) (Ctr 0 []) elems
-- BLOCK 239:
parseName :: ParserM String
parseName = skip >> many (alphaNum <|> char '_' <|> char '$' <|> char '&')
-- BLOCK 240:
parseName1 :: ParserM String
parseName1 = skip >> many1 (alphaNum <|> char '_' <|> char '$' <|> char '&')
-- BLOCK 241:
parseDef :: ParserM (String, ((Bool, [(Bool, String)]), Core))
parseDef = do
  copy <- option False $ do
    try $ do
      consume "!"
      return True
  try $ do
    skip
    consume "@"
  name <- parseName
  args <- option [] $ do
    try $ string "("
    args <- many $ do
      closeWith ")"
      bang <- option False $ do
        try $ do
          consume "!"
          return True
      arg <- parseName
      let strict = bang || head arg == '&'
      return (strict, arg)
    consume ")"
    return args
  skip
  consume "="
  core <- parseCore
  return (name, ((copy,args), core))
-- BLOCK 242:
parseADT :: ParserM ()
parseADT = do
  try $ do
    skip
    consume "data"
  name <- parseName
  skip
  consume "{"
  constructors <- many parseADTCtr
  consume "}"
  let ctrCids = zip (map fst constructors) [0..]
  let ctrAris = zip (map fst constructors) (map (fromIntegral . length . snd) constructors)
  modifyState (\s -> s { parsedCtrToCid = MS.union (MS.fromList ctrCids) (parsedCtrToCid s),
                         parsedCtrToAri = MS.union (MS.fromList ctrAris) (parsedCtrToAri s) })
-- BLOCK 243:
parseADTCtr :: ParserM (String, [String])
parseADTCtr = do
  skip
  consume "#"
  name <- parseName
  fields <- option [] $ do
    try $ consume "{"
    fds <- many $ do
      closeWith "}"
      parseName
    skip
    consume "}"
    return fds
  skip
  return (name, fields)
-- BLOCK 244:
parseBook :: ParserM [(String, ((Bool, [(Bool,String)]), Core))]
parseBook = do
  skip
  many parseADT
  defs <- many parseDef
  skip
  eof
  return defs
-- BLOCK 245:
doParseCore :: String -> IO Core
doParseCore code = case runParser parseCore (ParserState MS.empty MS.empty 0) "" code of
  Right core -> do
    return $ core
  Left  err  -> do
    showParseError "" code err
    return $ Ref "⊥" 0 []
-- BLOCK 246:
doParseBook :: String -> IO Book 
doParseBook code = case runParser parseBookWithState (ParserState MS.empty MS.empty 0) "" code of
  Right (defs, st) -> do
    return $ createBook defs (parsedCtrToCid st) (parsedCtrToAri st)
  Left err -> do
    showParseError "" code err
    return $ Book MS.empty MS.empty MS.empty MS.empty MS.empty MS.empty
  where
    parseBookWithState :: ParserM ([(String, ((Bool,[(Bool,String)]), Core))], ParserState)
    parseBookWithState = do
      defs <- parseBook
      st <- getState
      return (defs, st)
-- BLOCK 247:
-- Helper Parsers
-- --------------
-- BLOCK 248:
consume :: String -> ParserM String
consume str = skip >> string str
-- BLOCK 249:
closeWith :: String -> ParserM ()
closeWith str = try $ do
  skip
  notFollowedBy (string str)
-- BLOCK 250:
skip :: ParserM ()
skip = skipMany (parseSpace <|> parseComment) where
  parseSpace = (try $ do
    space
    return ()) <?> "space"
  parseComment = (try $ do
    string "//"
    skipMany (noneOf "\n")
    char '\n'
    return ()) <?> "Comment"
-- BLOCK 251:
genFreshLabel :: ParserM Word64
genFreshLabel = do
  st <- getState
  let lbl = freshLabel st
  putState st { freshLabel = lbl + 1 }
  return $ lbl + 0x800000
-- BLOCK 252:
-- Adjusting
-- ---------
-- BLOCK 253:
createBook :: [(String, ((Bool,[(Bool,String)]), Core))] -> MS.Map String Word64 -> MS.Map String Int -> Book
createBook defs ctrToCid ctrToAri =
  let withPrims = \n2i -> MS.union n2i $ MS.fromList primitives
      nameToId' = withPrims $ MS.fromList $ zip (map fst defs) [0..]
      idToName' = MS.fromList $ map (\(k,v) -> (v,k)) $ MS.toList nameToId'
      idToFunc' = MS.fromList $ map (\(name, ((copy,args), core)) -> (mget nameToId' name, ((copy,args), lexify (setRefIds nameToId' core)))) defs
      idToLabs' = MS.fromList $ map (\(name, (_, core)) -> (mget nameToId' name, collectLabels core)) defs
  in Book idToFunc' idToName' idToLabs' nameToId' ctrToAri ctrToCid
-- BLOCK 254:
-- Adds the function id to Ref constructors
setRefIds :: MS.Map String Word64 -> Core -> Core
setRefIds fids term = case term of
  Var nam       -> Var nam
  Let m x v b   -> Let m x (setRefIds fids v) (setRefIds fids b)
  Lam x bod     -> Lam x (setRefIds fids bod)
  App f x       -> App (setRefIds fids f) (setRefIds fids x)
  Sup l x y     -> Sup l (setRefIds fids x) (setRefIds fids y)
  Dup l x y v b -> Dup l x y (setRefIds fids v) (setRefIds fids b)
  Ctr cid fds   -> Ctr cid (map (setRefIds fids) fds)
  Mat x mov css -> Mat (setRefIds fids x) (map (\ (k,v) -> (k, setRefIds fids v)) mov) (map (\ (ctr,fds,cs) -> (ctr, fds, setRefIds fids cs)) css)
  Op2 op x y    -> Op2 op (setRefIds fids x) (setRefIds fids y)
  U32 n         -> U32 n
  Chr c         -> Chr c
  Era           -> Era
  Ref nam _ arg -> case MS.lookup nam fids of
    Just fid -> Ref nam fid (map (setRefIds fids) arg)
    Nothing  -> unsafePerformIO $ do
      putStrLn $ "error:unbound-ref @" ++ nam
      exitFailure
-- BLOCK 255:
-- Collects all SUP/DUP labels used
collectLabels :: Core -> MS.Map Word64 ()
collectLabels term = case term of
  Var _               -> MS.empty
  U32 _               -> MS.empty
  Chr _               -> MS.empty
  Era                 -> MS.empty
  Ref _ _ args        -> MS.unions $ map collectLabels args
  Let _ _ val bod     -> MS.union (collectLabels val) (collectLabels bod)
  Lam _ bod           -> collectLabels bod
  App fun arg         -> MS.union (collectLabels fun) (collectLabels arg)
  Sup lab tm0 tm1     -> MS.insert lab () $ MS.union (collectLabels tm0) (collectLabels tm1)
  Dup lab _ _ val bod -> MS.insert lab () $ MS.union (collectLabels val) (collectLabels bod)
  Ctr _ fds           -> MS.unions $ map collectLabels fds
  Mat val mov css     -> MS.unions $ collectLabels val : map (collectLabels . snd) mov ++ map (\(_,_,bod) -> collectLabels bod) css
  Op2 _ x y           -> MS.union (collectLabels x) (collectLabels y)
-- BLOCK 256:
-- Gives unique names to lexically scoped vars, unless they start with '$'.
-- Example: `λx λt (t λx(x) x)` will read as `λx0 λt1 (t1 λx2(x2) x0)`.
lexify :: Core -> Core
lexify term = evalState (go term MS.empty) 0 where
  fresh :: String -> State Int String
  fresh nam@('$':_) = return $ nam
  fresh nam         = do i <- get; put (i+1); return $ nam++"$"++show i
-- BLOCK 257:
  extend :: String -> String -> MS.Map String String -> State Int (MS.Map String String)
  extend old@('$':_) new ctx = return $ ctx
  extend old         new ctx = return $ MS.insert old new ctx
-- BLOCK 258:
  go :: Core -> MS.Map String String -> State Int Core
-- BLOCK 259:
  go term ctx = case term of
-- BLOCK 260:
    Var nam -> 
      return $ Var (MS.findWithDefault nam nam ctx)
-- BLOCK 261:
    Ref nam fid arg -> do
      arg <- mapM (\x -> go x ctx) arg
      return $ Ref nam fid arg
-- BLOCK 262:
    Let mod nam val bod -> do
      val  <- go val ctx
      nam' <- fresh nam
      ctx  <- extend nam nam' ctx
      bod  <- go bod ctx
      return $ Let mod nam' val bod
-- BLOCK 263:
    Lam nam bod -> do
      nam' <- fresh nam
      ctx  <- extend nam nam' ctx
      bod  <- go bod ctx
      return $ Lam nam' bod
-- BLOCK 264:
    App fun arg -> do
      fun <- go fun ctx
      arg <- go arg ctx
      return $ App fun arg
-- BLOCK 265:
    Sup lab tm0 tm1 -> do
      tm0 <- go tm0 ctx
      tm1 <- go tm1 ctx
      return $ Sup lab tm0 tm1
-- BLOCK 266:
    Dup lab dp0 dp1 val bod -> do
      val  <- go val ctx
      dp0' <- fresh dp0
      dp1' <- fresh dp1
      ctx  <- extend dp0 dp0' ctx
      ctx  <- extend dp1 dp1' ctx
      bod  <- go bod ctx
      return $ Dup lab dp0' dp1' val bod
-- BLOCK 267:
    Ctr cid fds -> do
      fds <- mapM (\x -> go x ctx) fds
      return $ Ctr cid fds
-- BLOCK 268:
    Mat val mov css -> do
      val' <- go val ctx
      mov' <- forM mov $ \ (k,v) -> do
        k' <- fresh k
        v  <- go v ctx
        return $ (k', v)
      css' <- forM css $ \ (ctr,fds,bod) -> do
        fds' <- mapM fresh fds
        ctx  <- foldM (\ ctx (fd,fd') -> extend fd fd' ctx) ctx (zip fds fds')
        ctx  <- foldM (\ ctx ((k,_),(k',_)) -> extend k k' ctx) ctx (zip mov mov')
        bod <- go bod ctx
        return (ctr, fds', bod)
      return $ Mat val' mov' css'
-- BLOCK 269:
    Op2 op nm0 nm1 -> do
      nm0 <- go nm0 ctx
      nm1 <- go nm1 ctx
      return $ Op2 op nm0 nm1
-- BLOCK 270:
    U32 n -> 
      return $ U32 n
-- BLOCK 271:
    Chr c ->
      return $ Chr c
-- BLOCK 272:
    Era -> 
      return Era
-- BLOCK 273:
-- Errors
-- ------
-- BLOCK 274:
-- Error handling
extractExpectedTokens :: ParseError -> String
extractExpectedTokens err =
    let expectedMsgs = [msg | Expect msg <- errorMessages err, msg /= "space", msg /= "Comment"]
    in intercalate " | " expectedMsgs
-- BLOCK 275:
showParseError :: String -> String -> ParseError -> IO ()
showParseError filename input err = do
  let pos = errorPos err
  let lin = sourceLine pos
  let col = sourceColumn pos
  let errorMsg = extractExpectedTokens err
  putStrLn $ setSGRCode [SetConsoleIntensity BoldIntensity] ++ "\nPARSE_ERROR" ++ setSGRCode [Reset]
  putStrLn $ "- expected: " ++ errorMsg
  putStrLn $ "- detected:"
  putStrLn $ highlightError (lin, col) (lin, col + 1) input
  putStrLn $ setSGRCode [SetUnderlining SingleUnderline] ++ filename ++ setSGRCode [Reset]

-- BLOCK 276:
-- BLOCK 277:
module HVML.Reduce where
-- BLOCK 278:
import Control.Monad (when, forM, forM_)
import Data.Word
import HVML.Collapse
import HVML.Extract
import HVML.Inject
import HVML.Show
import HVML.Type
import System.Exit
import qualified Data.Map.Strict as MS
-- BLOCK 279:
reduceAt :: Bool -> ReduceAt
-- BLOCK 280:
reduceAt debug book host = do 
  term <- got host
  let tag = termTag term
  let lab = termLab term
  let loc = termLoc term
-- BLOCK 281:
  when debug $ do
    root <- doExtractCoreAt (const got) book 0
    core <- doExtractCoreAt (const got) book host
    putStrLn $ "reduce: " ++ termToString term
    -- putStrLn $ "---------------- CORE: "
    -- putStrLn $ coreToString core
    putStrLn $ "---------------- ROOT: "
    putStrLn $ coreToString (doLiftDups root)
-- BLOCK 282:
  case tagT tag of
-- BLOCK 283:
    LET -> do
      case modeT lab of
        LAZY -> do
          val <- got (loc + 0)
          cont host (reduceLet term val)
        STRI -> do
          val <- reduceAt debug book (loc + 0)
          cont host (reduceLet term val)
        PARA -> do
          error "TODO"
-- BLOCK 284:
    APP -> do
      fun <- reduceAt debug book (loc + 0)
      case tagT (termTag fun) of
        ERA -> cont host (reduceAppEra term fun)
        LAM -> cont host (reduceAppLam term fun)
        SUP -> cont host (reduceAppSup term fun)
        CTR -> cont host (reduceAppCtr term fun)
        W32 -> cont host (reduceAppW32 term fun)
        CHR -> cont host (reduceAppW32 term fun)
        _   -> set (loc + 0) fun >> return term
-- BLOCK 285:
    MAT -> do
      val <- reduceAt debug book (loc + 0)
      case tagT (termTag val) of
        ERA -> cont host (reduceMatEra term val)
        LAM -> cont host (reduceMatLam term val)
        SUP -> cont host (reduceMatSup term val)
        CTR -> cont host (reduceMatCtr term val)
        W32 -> cont host (reduceMatW32 term val)
        CHR -> cont host (reduceMatW32 term val)
        _   -> set (loc + 0) val >> return term
-- BLOCK 286:
    OPX -> do
      val <- reduceAt debug book (loc + 0)
      case tagT (termTag val) of
        ERA -> cont host (reduceOpxEra term val)
        LAM -> cont host (reduceOpxLam term val)
        SUP -> cont host (reduceOpxSup term val)
        CTR -> cont host (reduceOpxCtr term val)
        W32 -> cont host (reduceOpxW32 term val)
        CHR -> cont host (reduceOpxW32 term val)
        _   -> set (loc + 0) val >> return term
-- BLOCK 287:
    OPY -> do
      val <- reduceAt debug book (loc + 1)
      case tagT (termTag val) of
        ERA -> cont host (reduceOpyEra term val)
        LAM -> cont host (reduceOpyLam term val)
        SUP -> cont host (reduceOpySup term val)
        CTR -> cont host (reduceOpyCtr term val)
        W32 -> cont host (reduceOpyW32 term val)
        CHR -> cont host (reduceOpyW32 term val)
        _   -> set (loc + 1) val >> return term
-- BLOCK 288:
    DP0 -> do
      sb0 <- got (loc + 0)
      if termGetBit sb0 == 0
        then do
          val <- reduceAt debug book (loc + 0)
          case tagT (termTag val) of
            ERA -> cont host (reduceDupEra term val)
            LAM -> cont host (reduceDupLam term val)
            SUP -> cont host (reduceDupSup term val)
            CTR -> cont host (reduceDupCtr term val)
            W32 -> cont host (reduceDupW32 term val)
            CHR -> cont host (reduceDupW32 term val)
            _   -> set (loc + 0) val >> return term
        else do
          set host (termRemBit sb0)
          reduceAt debug book host
-- BLOCK 289:
    DP1 -> do
      sb1 <- got (loc + 1)
      if termGetBit sb1 == 0
        then do
          val <- reduceAt debug book (loc + 0)
          case tagT (termTag val) of
            ERA -> cont host (reduceDupEra term val)
            LAM -> cont host (reduceDupLam term val)
            SUP -> cont host (reduceDupSup term val)
            CTR -> cont host (reduceDupCtr term val)
            W32 -> cont host (reduceDupW32 term val)
            CHR -> cont host (reduceDupW32 term val)
            _   -> set (loc + 0) val >> return term
        else do
          set host (termRemBit sb1)
          reduceAt debug book host
-- BLOCK 290:
    VAR -> do
      sub <- got (loc + 0)
      if termGetBit sub == 0
        then return term
        else do
          set host (termRemBit sub)
          reduceAt debug book host
-- BLOCK 291:
    REF -> do
      reduceRefAt book host
      reduceAt debug book host
-- BLOCK 292:
    otherwise -> do
      return term
-- BLOCK 293:
  where
    cont host action = do
      ret <- action
      set host ret
      reduceAt debug book host
-- BLOCK 294:
reduceRefAt :: Book -> Loc -> HVM Term
reduceRefAt book host = do
  term <- got host
  let lab = termLab term
  let loc = termLoc term
  let fid = u12v2X lab
  let ari = u12v2Y lab
  case fid of
    x | x == _DUP_F_ -> reduceRefAt_DupF book host loc ari
    x | x == _SUP_F_ -> reduceRefAt_SupF book host loc ari
    x | x == _LOG_F_ -> reduceRefAt_LogF book host loc ari
    x | x == _FRESH_F_ -> reduceRefAt_FreshF book host loc ari
    oterwise -> case MS.lookup fid (idToFunc book) of
      Just ((copy, args), core) -> do
        incItr
        when (length args /= fromIntegral ari) $ do
          putStrLn $ "RUNTIME_ERROR: arity mismatch on call to '@" ++ mget (idToName book) fid ++ "'."
          exitFailure
        argTerms <- if ari == 0
          then return [] 
          else forM (zip [0..] args) $ \(i, (strict, _)) -> do
            term <- got (loc + i)
            if strict
              then reduceAt False book (loc + i)
              else return term
        doInjectCoreAt book core host $ zip (map snd args) argTerms
        -- TODO: I disabled Fast Copy Optimization on interpreted mode because I
        -- don't think it is relevant here. We use it for speed, to trigger the
        -- hot paths on compiled functions, which don't happen when interpreted.
        -- I think leaving it out is good because it ensures interpreted mode is
        -- always optimal (minimizing interactions). This also allows the dev to
        -- see how Fast Copy Mode affects the interaction count.
        -- let inject = doInjectCoreAt book core host $ zip (map snd args) argTerms
        -- Fast Copy Optimization
        -- if copy then do
          -- let supGet = \x (idx,sup) -> if tagT (termTag sup) == SUP then Just (idx,sup) else x
          -- let supGot = foldl' supGet Nothing $ zip [0..] argTerms
          -- case supGot of
            -- Just (idx,sup) -> do
              -- let isCopySafe = case MS.lookup fid (idToLabs book) of
                    -- Nothing   -> False
                    -- Just labs -> not $ MS.member (termLab sup) labs
              -- if isCopySafe then do
                -- term <- reduceRefSup term idx
                -- set host term
                -- return term
              -- else inject
            -- otherwise -> inject
        -- else inject
      Nothing -> do
        return term
-- BLOCK 295:
-- Primitive: Dynamic Dup `@DUP(lab val λdp0λdp1(bod))`
reduceRefAt_DupF :: Book -> Loc -> Loc -> Word64 -> HVM Term  
reduceRefAt_DupF book host loc ari = do
  incItr
  when (ari /= 3) $ do
    putStrLn $ "RUNTIME_ERROR: arity mismatch on call to '@DUP'."
    exitFailure
  lab <- reduceAt False book (loc + 0)
  val <- got (loc + 1)
  bod <- got (loc + 2)
  dup <- allocNode 2
  case tagT (termTag lab) of
    W32 -> do
      when (termLoc lab >= 0x1000000) $ do
        error "RUNTIME_ERROR: dynamic DUP label too large"
      -- Create the DUP node with value and SUB
      set (dup + 0) val
      set (dup + 1) (termNew _SUB_ 0 0)
      -- Create first APP node for (APP bod DP0)
      app1 <- allocNode 2
      set (app1 + 0) bod
      set (app1 + 1) (termNew _DP0_ (termLoc lab) dup)
      -- Create second APP node for (APP (APP bod DP0) DP1)
      app2 <- allocNode 2
      set (app2 + 0) (termNew _APP_ 0 app1)
      set (app2 + 1) (termNew _DP1_ (termLoc lab) dup)
      let ret = termNew _APP_ 0 app2
      set host ret
      return ret
    _ -> do
      core <- doExtractCoreAt (\ x -> got) book (loc + 0)
      putStrLn $ "RUNTIME_ERROR: dynamic DUP without numeric label: " ++ termToString lab
      putStrLn $ coreToString (doLiftDups core)
      exitFailure
-- BLOCK 296:
-- Primitive: Dynamic Sup `@SUP(lab tm0 tm1)`
reduceRefAt_SupF :: Book -> Loc -> Loc -> Word64 -> HVM Term
reduceRefAt_SupF book host loc ari = do
  incItr
  when (ari /= 3) $ do
    putStrLn $ "RUNTIME_ERROR: arity mismatch on call to '@SUP'."
    exitFailure
  lab <- reduceAt False book (loc + 0)
  tm0 <- got (loc + 1)
  tm1 <- got (loc + 2)
  sup <- allocNode 2
  case tagT (termTag lab) of
    W32 -> do
      when (termLoc lab >= 0x1000000) $ do
        error "RUNTIME_ERROR: dynamic SUP label too large"
      let ret = termNew _SUP_ (termLoc lab) sup
      set (sup + 0) tm0
      set (sup + 1) tm1
      set host ret
      return ret
    _ -> error "RUNTIME_ERROR: dynamic SUP without numeric label."
-- BLOCK 297:
-- Primitive: Logger `@LOG(msg)`
-- Will extract the term and log it. 
-- Returns 0.
reduceRefAt_LogF :: Book -> Loc -> Loc -> Word64 -> HVM Term
reduceRefAt_LogF book host loc ari = do
  incItr
  when (ari /= 1) $ do
    putStrLn $ "RUNTIME_ERROR: arity mismatch on call to '@LOG'."
    exitFailure
  msg <- doExtractCoreAt (const got) book (loc + 0)
  putStrLn $ coreToString (doLiftDups msg)
  -- msgs <- doCollapseFlatAt (const got) book (loc + 0)
  -- forM_ msgs $ \msg -> do
    -- putStrLn $ coreToString msg
  let ret = termNew _W32_ 0 0
  set host ret
  return ret
-- BLOCK 298:
-- Primitive: Fresh `@FRESH`
-- Returns a fresh dup label.
reduceRefAt_FreshF :: Book -> Loc -> Loc -> Word64 -> HVM Term
reduceRefAt_FreshF book host loc ari = do
  incItr
  when (ari /= 0) $ do
    putStrLn $ "RUNTIME_ERROR: arity mismatch on call to '@Fresh'."
    exitFailure
  num <- fresh
  let ret = termNew _W32_ 0 num
  set host ret
  return ret
-- BLOCK 299:
reduceCAt :: Bool -> ReduceAt
reduceCAt = \ _ _ host -> do
  term <- got host
  whnf <- reduceC term
  set host whnf
  return $ whnf
-- BLOCK 300:
-- normalAtWith :: (Book -> Term -> HVM Term) -> Book -> Loc -> HVM Term
-- normalAtWith reduceAt book host = do
  -- term <- got host
  -- if termBit term == 1 then do
    -- return term
  -- else do
    -- whnf <- reduceAt book host
    -- set host $ termSetBit whnf
    -- let tag = termTag whnf
    -- let lab = termLab whnf
    -- let loc = termLoc whnf
    -- case tagT tag of
      -- APP -> do
        -- normalAtWith reduceAt book (loc + 0)
        -- normalAtWith reduceAt book (loc + 1)
        -- return whnf
      -- LAM -> do
        -- normalAtWith reduceAt book (loc + 1)
        -- return whnf
      -- SUP -> do
        -- normalAtWith reduceAt book (loc + 0)
        -- normalAtWith reduceAt book (loc + 1)
        -- return whnf
      -- DP0 -> do
        -- normalAtWith reduceAt book (loc + 0)
        -- return whnf
      -- DP1 -> do
        -- normalAtWith reduceAt book (loc + 0)
        -- return whnf
      -- CTR -> do
        -- let ari = u12v2Y lab
        -- let ars = (if ari == 0 then [] else [0 .. ari - 1]) :: [Word64]
        -- mapM_ (\i -> normalAtWith reduceAt book (loc + i)) ars
        -- return whnf
      -- MAT -> do
        -- let ari = lab
        -- let ars = [0 .. ari] :: [Word64]
        -- mapM_ (\i -> normalAtWith reduceAt book (loc + i)) ars
        -- return whnf
      -- _ -> do
        -- return whnf
-- BLOCK 301:
-- normalAt :: Book -> Loc -> HVM Term
-- normalAt = normalAtWith (reduceAt False)
-- BLOCK 302:
-- normalCAt :: Book -> Loc -> HVM Term
-- normalCAt = normalAtWith (reduceCAt False)

-- BLOCK 424:
-- BLOCK 425:
module HVML.Show where
-- BLOCK 426:
import Control.Applicative ((<|>))
import Control.Monad.State
import Data.Char (chr, ord)
import Data.Char (intToDigit)
import Data.IORef
import Data.List
import Data.Word
import HVML.Type
import Numeric (showIntAtBase)
import System.IO.Unsafe (unsafePerformIO)
import qualified Data.Map.Strict as MS
-- BLOCK 427:
-- Core Stringification
-- --------------------
-- BLOCK 428:
showCore :: Core -> String
showCore = coreToString . prettyRename
-- BLOCK 429:
coreToString :: Core -> String
coreToString core =
-- BLOCK 430:
  case pretty core of
    Just str -> str
    Nothing -> case core of
-- BLOCK 431:
      Var nam ->
        nam
-- BLOCK 432:
      Era ->
        "*"
-- BLOCK 433:
      Lam vr0 bod ->
        let bod' = coreToString bod in
        "λ" ++ vr0 ++ " " ++ bod'
-- BLOCK 434:
      App fun arg ->
        let fun' = coreToString fun in
        let arg' = coreToString arg in
        "(" ++ fun' ++ " " ++ arg' ++ ")"
-- BLOCK 435:
      Sup lab tm0 tm1 ->
        let tm0' = coreToString tm0 in
        let tm1' = coreToString tm1 in
        "&" ++ show lab ++ "{" ++ tm0' ++ " " ++ tm1' ++ "}"
-- BLOCK 436:
      Dup lab dp0 dp1 val bod ->
        let val' = coreToString val in
        let bod' = coreToString bod in
        "! &" ++ show lab ++ "{" ++ dp0 ++ " " ++ dp1 ++ "} = " ++ val' ++ "\n" ++ bod'
-- BLOCK 437:
      Ref nam fid arg ->
        let arg' = intercalate " " (map coreToString arg) in
        "@" ++ nam ++ "(" ++ arg' ++ ")"
-- BLOCK 438:
      Ctr cid fds ->
        let fds' = unwords (map coreToString fds) in
        "#" ++ show cid ++ "{" ++ fds' ++ "}"
-- BLOCK 439:
      Mat val mov css ->
        let val' = coreToString val in
        let mov' = concatMap (\ (k,v) -> " !" ++ k ++ "=" ++ coreToString v) mov in
        let css' = unwords [ctr ++ "{" ++ unwords fds ++ "}:" ++ coreToString bod | (ctr, fds, bod) <- css] in
        "(~" ++ val' ++ mov' ++ " {" ++ css' ++ "})"
-- BLOCK 440:
      U32 val ->
        show val
-- BLOCK 441:
      Chr val ->
        "'" ++ [val] ++ "'"
-- BLOCK 442:
      Op2 opr nm0 nm1 ->
        let nm0' = coreToString nm0 in
        let nm1' = coreToString nm1 in
        "(" ++ operToString opr ++ " " ++ nm0' ++ " " ++ nm1' ++ ")"
-- BLOCK 443:
      Let mod nam val bod ->
        if nam == "" then
          let val' = coreToString val in
          let bod' = coreToString bod in
          val' ++ "\n" ++ bod'
        else
          let val' = coreToString val in
          let bod' = coreToString bod in
          "! " ++ modeToString mod ++ nam ++ " = " ++ val' ++ "\n" ++ bod'
-- BLOCK 444:
operToString :: Oper -> String
operToString OP_ADD = "+"
operToString OP_SUB = "-"
operToString OP_MUL = "*"
operToString OP_DIV = "/"
operToString OP_MOD = "%"
operToString OP_EQ  = "=="
operToString OP_NE  = "!="
operToString OP_LT  = "<"
operToString OP_GT  = ">"
operToString OP_LTE = "<="
operToString OP_GTE = ">="
operToString OP_AND = "&"
operToString OP_OR  = "|"
operToString OP_XOR = "^"
operToString OP_LSH = "<<"
operToString OP_RSH = ">>"
-- BLOCK 445:
modeToString LAZY = ""
modeToString STRI = "."
modeToString PARA = "^"
-- BLOCK 446:
-- Runtime Stringification
-- -----------------------
-- BLOCK 447:
tagToString :: Tag -> String
tagToString t = show (tagT t)
-- BLOCK 448:
labToString :: Word64 -> String
labToString loc = padLeft (showHex loc) 6 '0'
-- BLOCK 449:
locToString :: Word64 -> String
locToString loc = padLeft (showHex loc) 9 '0'
-- BLOCK 450:
termToString :: Term -> String
termToString term =
  let tag = tagToString (termTag term)
      lab = labToString (termLab term)
      loc = locToString (termLoc term)
  in "term_new(" ++ tag ++ ",0x" ++ lab ++ ",0x" ++ loc ++ ")"
-- BLOCK 451:
-- Pretty Renaming
-- ---------------
-- BLOCK 452:
prettyRename :: Core -> Core
prettyRename core = unsafePerformIO $ do
  namesRef <- newIORef MS.empty
  go namesRef core
  where
-- BLOCK 453:
    go namesRef core = case core of
-- BLOCK 454:
      Var name -> do
        name' <- genName namesRef name
        return $ Var name'
-- BLOCK 455:
      Lam name body -> do
        name' <- genName namesRef name
        body' <- go namesRef body
        return $ Lam name' body'
-- BLOCK 456:
      Let mode name val body -> do
        name' <- genName namesRef name
        val' <- go namesRef val
        body' <- go namesRef body
        return $ Let mode name' val' body'
-- BLOCK 457:
      App fun arg -> do
        fun' <- go namesRef fun
        arg' <- go namesRef arg
        return $ App fun' arg'
-- BLOCK 458:
      Sup lab x y -> do
        x' <- go namesRef x
        y' <- go namesRef y
        return $ Sup lab x' y'
-- BLOCK 459:
      Dup lab x y val body -> do
        x' <- genName namesRef x
        y' <- genName namesRef y
        val' <- go namesRef val
        body' <- go namesRef body
        return $ Dup lab x' y' val' body'
-- BLOCK 460:
      Ctr cid args -> do
        args' <- mapM (go namesRef) args
        return $ Ctr cid args'
-- BLOCK 461:
      Mat val mov css -> do
        val' <- go namesRef val
        mov' <- mapM (\(k,v) -> do v' <- go namesRef v; return (k,v')) mov
        css' <- mapM (\(c,vs,t) -> do t' <- go namesRef t; return (c,vs,t')) css
        return $ Mat val' mov' css'
-- BLOCK 462:
      Op2 op x y -> do
        x' <- go namesRef x
        y' <- go namesRef y
        return $ Op2 op x' y'
-- BLOCK 463:
      Ref name fid args -> do
        args' <- mapM (go namesRef) args
        return $ Ref name fid args'
-- BLOCK 464:
      other -> return other
-- BLOCK 465:
    genName namesRef name = do
      nameMap <- readIORef namesRef
      case MS.lookup name nameMap of
        Just name' -> return name'
        Nothing -> do
          let newName = genNameFromIndex (MS.size nameMap)
          modifyIORef' namesRef (MS.insert name newName)
          return newName
-- BLOCK 466:
    genNameFromIndex n = go (n + 1) "" where
      go n ac | n == 0    = ac
              | otherwise = go q (chr (ord 'a' + r) : ac)
              where (q,r) = quotRem (n - 1) 26
-- BLOCK 467:
-- Pretty Printers
-- ---------------
-- BLOCK 468:
pretty :: Core -> Maybe String
pretty core = prettyStr core <|> prettyLst core
-- pretty core = prettyStr core
-- BLOCK 469:
prettyStr :: Core -> Maybe String
prettyStr (Ctr 0 []) = Just "\"\""
prettyStr (Ctr 1 [Chr h, t]) = do
  rest <- prettyStr t
  return $ "\"" ++ h : tail rest
prettyStr _ = Nothing
-- BLOCK 470:
prettyLst :: Core -> Maybe String
prettyLst (Ctr 0 []) = Just "[]"
prettyLst (Ctr 1 [x, xs]) = do
  rest <- prettyLst xs
  return $ "[" ++ coreToString x ++ if rest == "[]" then "]" else " " ++ tail rest
prettyLst _ = Nothing
-- BLOCK 471:
-- Dumping
-- -------
-- BLOCK 472:
dumpHeapRange :: Word64 -> Word64 -> HVM [(Word64, Term)]
dumpHeapRange ini len =
  if ini < len then do
    head <- got ini
    tail <- dumpHeapRange (ini + 1) len
    if head == 0
      then return tail
      else return ((ini, head) : tail)
  else return []
-- BLOCK 473:
dumpHeap :: HVM ([(Word64, Term)], Word64)
dumpHeap = do
  len <- getLen
  itr <- getItr
  terms <- dumpHeapRange 0 len
  return (terms, itr)
-- BLOCK 474:
heapToString :: ([(Word64, Term)], Word64) -> String
heapToString (terms, itr) = 
  "set_itr(0x" ++ padLeft (showHex itr) 9 '0' ++ ");\n" ++
  foldr (\(k,v) txt ->
    let addr = padLeft (showHex k) 9 '0'
        term = termToString v
    in "set(0x" ++ addr ++ ", " ++ term ++ ");\n" ++ txt) "" terms
-- BLOCK 475:
padLeft :: String -> Int -> Char -> String
padLeft str n c = replicate (n - length str) c ++ str
-- BLOCK 476:
showHex :: Word64 -> String
showHex x = showIntAtBase 16 intToDigit (fromIntegral x) ""

-- BLOCK 477:
module HVML.Type where
-- BLOCK 478:
import Data.Map.Strict as MS
import Data.Word
import Foreign.Ptr
-- BLOCK 479:
-- Core Types
-- ----------
-- BLOCK 480:
--show--
data Core
  = Var String -- x
  | Ref String Word64 [Core] -- @fn
  | Era -- *
  | Lam String Core -- λx(F)
  | App Core Core -- (f x)
  | Sup Word64 Core Core -- &L{a b}
  | Dup Word64 String String Core Core -- ! &L{a b} = v body
  | Ctr Word64 [Core] -- #Ctr{a b ...}
  | Mat Core [(String,Core)] [(String,[String],Core)] -- ~ v { #A{a b ...}: ... #B{a b ...}: ... ... }
  | U32 Word32 -- 123
  | Chr Char -- 'a'
  | Op2 Oper Core Core -- (+ a b)
  | Let Mode String Core Core -- ! x = v body
  deriving (Show, Eq)
-- BLOCK 481:
--show--
data Mode
  = LAZY
  | STRI
  | PARA
  deriving (Show, Eq, Enum)
-- BLOCK 482:
--show--
data Oper
  = OP_ADD | OP_SUB | OP_MUL | OP_DIV
  | OP_MOD | OP_EQ  | OP_NE  | OP_LT
  | OP_GT  | OP_LTE | OP_GTE | OP_AND
  | OP_OR  | OP_XOR | OP_LSH | OP_RSH
  deriving (Show, Eq, Enum)
-- BLOCK 483:
--show--
-- A top-level function, including:
-- - copy: true when ref-copy mode is enabled
-- - args: a list of (isArgStrict, argName) pairs
-- - core: the function's body
-- Note: ref-copy improves C speed, but increases interaction count
type Func = ((Bool, [(Bool,String)]), Core)
-- BLOCK 484:
--show--
-- NOTE: the new idToLabs field is a map from a function id to a set of all
-- DUP/SUP labels used in its body. note that, when a function uses either
-- HVM.SUP or HVM.DUP internally, this field is set to Nothing. this will be
-- used to apply the fast DUP-REF and REF-SUP interactions, when safe to do so
data Book = Book
  { idToFunc :: MS.Map Word64 Func
  , idToName :: MS.Map Word64 String
  , idToLabs :: MS.Map Word64 (MS.Map Word64 ())
  , nameToId :: MS.Map String Word64
  , ctrToAri :: MS.Map String Int
  , ctrToCid :: MS.Map String Word64
  } deriving (Show, Eq)
-- BLOCK 485:
-- Runtime Types
-- -------------
-- BLOCK 486:
--show--
type Tag  = Word64
type Lab  = Word64
type Loc  = Word64
type Term = Word64
-- BLOCK 487:
--show--
data TAG
  = DP0
  | DP1
  | VAR
  | ERA
  | APP
  | LAM
  | SUP
  | SUB
  | REF
  | LET
  | CTR
  | MAT
  | W32
  | CHR
  | OPX
  | OPY
  deriving (Eq, Show)
-- BLOCK 488:
--show--
type HVM = IO
-- BLOCK 489:
--show--
type ReduceAt = Book -> Loc -> HVM Term
-- BLOCK 490:
-- C Functions
-- -----------
-- BLOCK 491:
foreign import ccall unsafe "Runtime.c hvm_init"
  hvmInit :: IO ()
foreign import ccall unsafe "Runtime.c hvm_free"
  hvmFree :: IO ()
foreign import ccall unsafe "Runtime.c alloc_node"
  allocNode :: Word64 -> IO Word64
foreign import ccall unsafe "Runtime.c set"
  set :: Word64 -> Term -> IO ()
foreign import ccall unsafe "Runtime.c got"
  got :: Word64 -> IO Term
foreign import ccall unsafe "Runtime.c take"
  take :: Word64 -> IO Term
foreign import ccall unsafe "Runtime.c swap"
  swap :: Word64 -> IO Term
foreign import ccall unsafe "Runtime.c term_new"
  termNew :: Tag -> Lab -> Loc -> Term
foreign import ccall unsafe "Runtime.c term_tag"
  termTag :: Term -> Tag
foreign import ccall unsafe "Runtime.c term_get_bit"
  termGetBit :: Term -> Tag
foreign import ccall unsafe "Runtime.c term_lab"
  termLab :: Term -> Lab
foreign import ccall unsafe "Runtime.c term_loc"
  termLoc :: Term -> Loc
foreign import ccall unsafe "Runtime.c term_set_bit"
  termSetBit :: Term -> Tag
foreign import ccall unsafe "Runtime.c term_rem_bit"
  termRemBit :: Term -> Tag
foreign import ccall unsafe "Runtime.c get_len"
  getLen :: IO Word64
foreign import ccall unsafe "Runtime.c get_itr"
  getItr :: IO Word64
foreign import ccall unsafe "Runtime.c inc_itr"
  incItr :: IO Word64
foreign import ccall unsafe "Runtime.c fresh"
  fresh :: IO Word64
foreign import ccall unsafe "Runtime.c reduce"
  reduceC :: Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_let"
  reduceLet :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_app_era"
  reduceAppEra :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_app_lam"
  reduceAppLam :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_app_sup"
  reduceAppSup :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_app_ctr"
  reduceAppCtr :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_app_w32"
  reduceAppW32 :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_dup_era"
  reduceDupEra :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_dup_lam"
  reduceDupLam :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_dup_sup"
  reduceDupSup :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_dup_ctr"
  reduceDupCtr :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_dup_w32"
  reduceDupW32 :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_dup_ref"
  reduceDupRef :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_mat_era"
  reduceMatEra :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_mat_lam"
  reduceMatLam :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_mat_sup"
  reduceMatSup :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_mat_ctr"
  reduceMatCtr :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_mat_w32"
  reduceMatW32 :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opx_era"
  reduceOpxEra :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opx_lam"
  reduceOpxLam :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opx_sup"
  reduceOpxSup :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opx_ctr"
  reduceOpxCtr :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opx_w32"
  reduceOpxW32 :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opy_era"
  reduceOpyEra :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opy_lam"
  reduceOpyLam :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opy_sup"
  reduceOpySup :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opy_ctr"
  reduceOpyCtr :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_opy_w32"
  reduceOpyW32 :: Term -> Term -> IO Term
foreign import ccall unsafe "Runtime.c reduce_ref_sup"
  reduceRefSup :: Term -> Word64 -> IO Term
foreign import ccall unsafe "Runtime.c hvm_define"
  hvmDefine :: Word64 -> FunPtr (IO Term) -> IO ()
foreign import ccall unsafe "Runtime.c hvm_get_state"
  hvmGetState :: IO (Ptr ())
foreign import ccall unsafe "Runtime.c hvm_set_state"
  hvmSetState :: Ptr () -> IO ()
foreign import ccall unsafe "Runtime.c u12v2_new"
  u12v2New :: Word64 -> Word64 -> Word64
foreign import ccall unsafe "Runtime.c u12v2_x"
  u12v2X :: Word64 -> Word64
foreign import ccall unsafe "Runtime.c u12v2_y"
  u12v2Y :: Word64 -> Word64
-- BLOCK 492:
-- Constants
-- ---------
-- BLOCK 493:
--show--
tagT :: Tag -> TAG
tagT 0x00 = DP0
tagT 0x01 = DP1
tagT 0x02 = VAR
tagT 0x03 = SUB
tagT 0x04 = REF
tagT 0x05 = LET
tagT 0x06 = APP
tagT 0x08 = MAT
tagT 0x09 = OPX
tagT 0x0A = OPY
tagT 0x0B = ERA
tagT 0x0C = LAM
tagT 0x0D = SUP
tagT 0x0F = CTR
tagT 0x10 = W32
tagT 0x11 = CHR
tagT tag  = error $ "unknown tag: " ++ show tag
-- BLOCK 494:
_DP0_ :: Tag
_DP0_ = 0x00
-- BLOCK 495:
_DP1_ :: Tag
_DP1_ = 0x01
-- BLOCK 496:
_VAR_ :: Tag
_VAR_ = 0x02
-- BLOCK 497:
_SUB_ :: Tag
_SUB_ = 0x03
-- BLOCK 498:
_REF_ :: Tag
_REF_ = 0x04
-- BLOCK 499:
_LET_ :: Tag
_LET_ = 0x05
-- BLOCK 500:
_APP_ :: Tag
_APP_ = 0x06
-- BLOCK 501:
_MAT_ :: Tag
_MAT_ = 0x08
-- BLOCK 502:
_OPX_ :: Tag
_OPX_ = 0x09
-- BLOCK 503:
_OPY_ :: Tag
_OPY_ = 0x0A
-- BLOCK 504:
_ERA_ :: Tag
_ERA_ = 0x0B
-- BLOCK 505:
_LAM_ :: Tag
_LAM_ = 0x0C
-- BLOCK 506:
_SUP_ :: Tag
_SUP_ = 0x0D
-- BLOCK 507:
_CTR_ :: Tag
_CTR_ = 0x0F
-- BLOCK 508:
_W32_ :: Tag
_W32_ = 0x10
-- BLOCK 509:
_CHR_ :: Tag
_CHR_ = 0x11
-- BLOCK 510:
--show--
modeT :: Lab -> Mode
modeT 0x00 = LAZY
modeT 0x01 = STRI
modeT 0x02 = PARA
modeT mode = error $ "unknown mode: " ++ show mode
-- BLOCK 511:
-- Primitive Functions
_DUP_F_ :: Lab
_DUP_F_ = 0xFFF
-- BLOCK 512:
_SUP_F_ :: Lab
_SUP_F_ = 0xFFE
-- BLOCK 513:
_LOG_F_ :: Lab
_LOG_F_ = 0xFFD
-- BLOCK 514:
_FRESH_F_ :: Lab
_FRESH_F_ = 0xFFC
-- BLOCK 515:
primitives :: [(String, Lab)]
primitives = 
  [ ("SUP", _SUP_F_)
  , ("DUP", _DUP_F_)
  , ("LOG", _LOG_F_)
  , ("FRESH", _FRESH_F_)
  ]
-- BLOCK 516:
-- Utils
-- -----
-- BLOCK 517:
-- Getter function for maps
mget map key =
  case MS.lookup key map of
    Just val -> val
    Nothing  -> error $ "key not found: " ++ show key
-- BLOCK 518:
-- The if-let match stores its target ctr id
ifLetLab :: Book -> Core -> Word64
ifLetLab book (Mat _ _ [(ctr,_,_),("_",_,_)]) =
  case MS.lookup ctr (ctrToCid book) of
    Just cid -> 1 + cid
    Nothing  -> 0
ifLetLab book _ = 0